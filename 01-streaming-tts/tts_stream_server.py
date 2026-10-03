"""
Cloned-voice TTS -- sentence-level pipeline / delayed-start streaming version (prototype, port 8124).

Why this exists:
  The production tts_server.py (:8123) returns only after synthesizing the whole WAV, so device first-sound latency = total synthesis time
  (~17.6s / two sentences). This file splits the reply into sentences and streams while synthesizing, so the device starts speaking as soon as
  the first sentence is done.

Key constraints (all from measurement/forensics; do not change on intuition. 1-5 established 2026-07-16, 6 on 2026-07-17):
  1. The device really does consume a stream: OpenAITTS::stream -> AudioFileSourceHttpsPostStream
     -> AudioFileSourceBuffer(30KB) -> playWAV plays as bytes arrive. Writing 0xFFFFFFFF into the WAV data-size
     lets AudioGeneratorWAV keep playing until stream EOF (connection close).
  2. The device reads the raw http.getStreamPtr() stream and does not decode chunked. So the response must be
     "close-delimited" (Connection: close + no Content-Length + no chunked); closing the socket
     marks the end. uvicorn/h11 sends chunked even with Connection: close (verified), hence the raw socket here.
  3. Device underrun tolerance is only ~1.1s (30KB buffer ~0.64s + http.end() disconnects permanently after a 500ms blocking read).
     Synthesis is ~1.8x slower than real time (measured), so sentence one cannot be sent the moment it is done, or the device drops before sentence two is ready.
     Remedy = delayed start: accumulate a headstart before sending the first byte; on a real underrun mid-stream, fill silence (degrade to a short pause,
     never cut off). The device waits until it receives response headers (POST timeout 65s), so sending nothing
     while accumulating the headstart is safe.
  4. Synthesis must be single-threaded (MLX GPU streams are thread-bound). Same as production: one dedicated synthesis thread, one segment at a time.
     What runs concurrently is "transmit/play" vs "synthesize the next segment", never two syntheses in parallel. The device and a second client
     share this lock.
  5. Sentence splitting must happen after sanitize() (the em dash fix lives there, one guard for both paths). English mode drops
     all non-ASCII; Chinese mode keeps Unicode and splits natural Chinese text after full-width stops (U+3002, U+FF1F, U+FF01) with no following space.
  6. Idle unload (TTS_IDLE_UNLOAD, default 600s): **`del model` is nowhere near enough** -- MLX holds GPU memory
     in its own cache pool and does not return it to the system. Measured 2026-07-17: footprint after a real synthesis 3691MB
     -> still 3472MB after `del`+gc -> 668MB after `mx.clear_cache()`. **clear_cache() must be called explicitly**
     to actually release it (~3.0GB); with only del, the feature "looks like it runs but saves not a single byte".
     Unload and reload must both happen on the synthesis thread (same thread binding as constraint 4).
     Cost: first sentence after unload **+~2.9s** (measured: reload itself 2.0s; first byte cold 6.26s vs warm 3.32s).
     ★ Do not trust the "0.8s" computed by the probe -- on 2026-07-17 I derived it from separate load/generate timings;
       measured end to end on the real device it is 2.9s. **Summed component timings != end-to-end measurement.**
     `TTS_IDLE_UNLOAD=0` disables it, reverting to the always-resident behavior of 07-16.
     Note: reload time must not enter observe_chunk's wall (see the _generation_thread comment), or it poisons the estimator.

Swappable synthesis backend (for deterministic local tests without loading a second model):
  When TTS_SIM_DIR points to a directory, the "replay" backend is used: look up <sha1>.pcm by the sha1 of the sentence text,
  sleep for the measured synthesis time, then return the real PCM. Streaming/framing/start/underrun logic can thus be verified locally with real audio and timing;
  the only thing replaced is the "MLX live audio" step (left to the on-device smoke test).
"""

import gc
import hashlib
import io
import json
import os
import queue
import re
import socket
import socketserver
import threading
import time
import wave

import numpy as np

from voice_clarity import filter_profile

# ---- Voice recipe ---------------------------------------------------------------
# With no environment variables set, this is still the original formal English voice. The Chinese Qwen3 service is enabled via a separate LaunchAgent/port
# and does not change the default behavior of the existing English service on 8123.
BACKEND = os.environ.get("TTS_BACKEND", "chatterbox").strip().lower()
_DEFAULT_MODEL_ID = (
    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    if BACKEND == "qwen3"
    else "mlx-community/chatterbox-fp16"
)
MODEL_ID = os.environ.get("TTS_MODEL_ID", _DEFAULT_MODEL_ID).strip()
LANG_CODE = os.environ.get("TTS_LANG_CODE", "en").strip().lower()
REF_AUDIO = os.environ.get(
    "TTS_REF_AUDIO",
    "/path/to/voice/ref_voice.wav",
)
REF_TEXT = os.environ.get("TTS_REF_TEXT", "").strip()
RECIPE = {
    "exaggeration": 0.1,
    "cfg_weight": 0.5,
    "temperature": 0.7,
    "lang_code": LANG_CODE,
}
QWEN_TEMPERATURES = tuple(
    float(value.strip())
    for value in os.environ.get("TTS_QWEN_TEMPERATURES", "0.7").split(",")
    if value.strip()
)
QWEN_MAX_CHARS = int(os.environ.get("TTS_QWEN_MAX_CHARS", "18"))
# Whole-sentence synthesis switch; see the notes in stream_utterance(). Default 0 = keep original behavior.
QWEN_MERGE = os.environ.get("TTS_QWEN_MERGE", "0").strip().lower() in ("1", "true", "yes")
# Speech-rate multiplier; see time_stretch(). Default 1.0 = unchanged; the English 8123 track is unaffected (only set in the Chinese plist).
SPEED_RATE = float(os.environ.get("TTS_SPEED_RATE", "1.0"))
# ★ 2026-07-25: 🔴 Do not try model.generate(speed=...) again -- I tried; it is a dead end.
# The docstring of that parameter in mlx_audio's qwen3_tts.py literally says "not directly supported yet";
# measured at 1.0/1.15/1.3/1.5, audio durations were 2.64/2.96/2.88/3.04s (up, not down -- pure temperature noise), RTF constant at 2.75.
# Worse, qwen3_tts.py:230 has a check `if speed not in (None, 1.0) ... return False`,
# so passing a non-1.0 value may actually disable a fast path. Rate can only be changed by pitch-preserving time stretching after synthesis (see time_stretch).
INTER_CHUNK_PAUSE_S = float(
    os.environ.get("TTS_INTER_CHUNK_PAUSE", "0.2" if BACKEND == "qwen3" else "0")
)   # ★ now the **upper bound** of the inter-segment pause, no longer a fixed value; see inter_chunk_pause()
PAUSE_RATIO = float(os.environ.get("TTS_PAUSE_RATIO", "0.06"))   # pause = estimated whole-sentence duration x this ratio
PAUSE_FLOOR_S = float(os.environ.get("TTS_PAUSE_FLOOR", "0.06")) # never shorter than this, so two segments do not run together

SAMPLE_RATE = 24000          # chatterbox output is fixed at 24k, mono 16-bit
BYTES_PER_SEC = SAMPLE_RATE * 2

HOST = "0.0.0.0"
# Default 8124 runs alongside production 8123; for on-device smoke tests, stop launchd first (to free the old model's memory), then take over with TTS_PORT=8123
PORT = int(os.environ.get("TTS_PORT", "8124"))
LOG_TAG = f"tts{PORT}"

# ---- Start / splitting parameters (constants are measured; see file header) --------------------------------
# Formal English voice measured: 44 chars -> 3.2s, 85 chars -> 6.1s. Chinese characters carry more information; the first listening test measured
# about 0.32s/char; calibrate independently with TTS_SEC_PER_CHAR without affecting voice or generation speed.
_DEFAULT_SEC_PER_CHAR = 0.14 if BACKEND == "qwen3" else 0.32 if LANG_CODE == "zh" else 0.072
SEC_PER_CHAR = float(os.environ.get("TTS_SEC_PER_CHAR", str(_DEFAULT_SEC_PER_CHAR)))
RTF = 1.8                    # synthesis wall time / audio duration (measured warm)
MIN_CHUNK_AUDIO = 2.6        # target minimum audio seconds per synthesis block (amortizes the ~4.5s fixed synthesis floor of short sentences)
SILENCE_FRAME = 0.15         # seconds of silence filled per underrun step
SIM_FLOOR_S = 4.5            # simulated backend: synthesis floor for very short sentences
SAFETY_S = 1.5               # initial margin errs safe (seamless from the first sentence, then shrinks on its own while clean); adaptively tuned
LOW_WATER = 0.5             # phase B: fill silence only when the estimated device buffer is below this (s); otherwise wait for the real block (no premature injection)

# ★ 2026-07-25: the **floor** of the margin, split out of LOW_WATER into its own constant.
#
# Why split: observe_request() used `max(LOW_WATER, margin-0.1)` to shrink the margin, which
# tied two unrelated things to the same 0.5: "the phase-B silence-fill trigger level" and "the floor of the start margin".
# Raising the floor would also move the silence-fill trigger: touch one, break two.
#
# Why raise it (measured; see 109 real requests in tts-zh-server.log):
#   12 requests underran (11%); the longest silence inserted mid-sentence was 2.1s -- this is the "stutter" the user reported.
#   Restoring the margin in stats to its value **at decision time** (note observe_request has a dead zone:
#   it only adds +0.4 when silence_filled>0.4; in the 0.1-0.4 cases the margin never moved):
#     0.5 0.8 0.5 0.5 0.5 0.9 0.9 0.5 0.5 1.3 1.2 1.4
#   of the 12, 6 decided with margin exactly =0.5 and 8 with <=0.9 -- i.e. they **stuttered sitting on the floor**.
#   The controller is designed to "probe down -0.1 when clean, back off +0.4 on a stutter", so it inevitably walks to the floor
#   and stutters against it repeatedly. The floor itself was set too low; the controller has no bug.
#
# Why 1.2: it covers 10 of the 12 and misses only the 1.3/1.4 ones -- their speeds were
# 0.97/1.03 (the estimator was accurate); they stalled elsewhere, which raising the floor cannot fix. Do not expect this change to clear everything.
#
# ⚠️ The dead zone itself is a separate problem (not touched here): 0.1-0.4s pauses are audible, yet the controller learns nothing,
# and the margin keeps walking down by -0.1. Fixing "small stutters that keep recurring" means changing the dead zone, not the floor.
# Cost: playback starts ~0.7s later (0.5->1.2). Trading "a bit slower" for "no stutter" -- the user's call on 07-25.
#
# 🔴 This .py is **shared** by 8123 (English/chatterbox) and 8125 (Chinese/qwen3). The English track's
# RTF is only 1.5-1.8 with enough headroom and no underrun records -- so defaults are split by backend,
# raising only qwen3 and leaving English at its original 0.5. Before changing this, make sure it will not hurt 8123.
MARGIN_FLOOR_S = float(
    os.environ.get("TTS_MARGIN_FLOOR", "1.2" if BACKEND == "qwen3" else "0.5")
)

# Start policy: smooth = delay to the earliest point where "every remaining block arrives in time" -> no pauses (gap-free);
#           early  = start as soon as the first block is ready -> earliest first sound, but blocks may have silent gaps between them (backstopped by underrun silence fill).
START_POLICY = os.environ.get("TTS_START", "smooth").lower()

# ---- Idle unload (added 2026-07-17; see header constraint 6) -----------------------------
# Idle longer than this -> unload the model and release GPU memory; 0 = disabled (always resident, i.e. the 07-16 behavior).
# Measured (2026-07-17, this machine): resident footprint 3691MB / IOAccelerator 3172MB;
#   after unload 668MB / 7MB  =>  net reclaim ~3.0GB (19% of a 16GB machine).
# Cost: first sentence after unload +~2.9s. Measured on device (2026-07-17): reload itself 2.0s;
#   first byte cold (incl. reload) 6.26s vs warm 3.32s. => a two-sentence reply goes from 8.2s to about 11s when cold.
# ★ Do not trust "0.8s": it was derived from the probe's separate load/generate timings; end to end it is 2.9s.
IDLE_UNLOAD_S = int(os.environ.get("TTS_IDLE_UNLOAD", "600"))

# Output voice post-processing: off / natural / bright. Changes only frequency response and dynamics, not speaker, pitch, rate, or duration.
CLARITY_PROFILE = os.environ.get("TTS_CLARITY", "off").lower()
OUTPUT_GAIN = float(os.environ.get("TTS_OUTPUT_GAIN", "1.0"))

_PUNCT_MAP = {
    "—": ", ", "–": ", ", "‒": ", ", "…": ", ",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "‐": "-", "‑": "-", " ": " ",
}


def sanitize(text: str) -> str:
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    if LANG_CODE == "en":
        text = "".join(ch for ch in text if ord(ch) < 128)
    else:
        text = "".join(
            ch if ch.isprintable() else " " if ch.isspace() else "" for ch in text
        )
    return " ".join(text.split())


# ---- Sentence splitting + blocking -----------------------------------------------------------
# Chinese punctuation is usually not followed by a space; English still requires one, so things like 3.14 are not split.
_SENT_SPLIT = re.compile(r"(?<=[\u3002\uff1f\uff01])|(?<=[.?!])\s+")
_QWEN_SEGMENT_SPLIT = re.compile(r"(?<=[\u3002\uff1f\uff01\uff1b;\uff0c,])|(?<=[.?!])\s+")


def split_sentences(text: str) -> list:
    """Split sentences after sanitize. Fragments that are too short (<5 chars) merge into the next sentence (a trailing fragment merges into the previous one)."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    if not parts:
        return []
    merged, carry = [], ""
    for p in parts:
        p = (carry + " " + p).strip() if carry else p
        if len(p) < 5:
            carry = p
        else:
            merged.append(p)
            carry = ""
    if carry:
        if merged:
            merged[-1] = merged[-1] + " " + carry
        else:
            merged.append(carry)
    return merged


def split_qwen_segments(text: str) -> list:
    """Qwen3 generates from short spoken segments; split on commas too, and do not re-merge short segments."""
    segments = []
    for part in _QWEN_SEGMENT_SPLIT.split(text):
        part = part.strip()
        while len(part) > QWEN_MAX_CHARS:
            cut = QWEN_MAX_CHARS
            whitespace = part.rfind(" ", 1, QWEN_MAX_CHARS + 1)
            if whitespace >= max(4, QWEN_MAX_CHARS // 2):
                cut = whitespace
            segments.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            segments.append(part)
    return segments


def make_chunks(sentences: list) -> list:
    """Greedily merge sentences into blocks: the first block as small as possible (one sentence, early start); every other block accumulates >= MIN_CHUNK_AUDIO
    of estimated audio duration, amortizing the fixed floor cost of each synthesis."""
    if not sentences:
        return []
    chunks = [sentences[0]]          # first block = first sentence, aiming for the earliest start
    cur = ""
    for s in sentences[1:]:
        cur = (cur + " " + s).strip() if cur else s
        if len(cur) * SEC_PER_CHAR >= MIN_CHUNK_AUDIO:
            chunks.append(cur)
            cur = ""
    if cur:
        chunks.append(cur)
    return chunks


def est_audio_sec(text: str) -> float:
    return len(text) * SEC_PER_CHAR


# ---- Adaptive estimator (synth_wall ≈ speed * (FLOOR_NOM + SLOPE_NOM*audio)) --------
# speed is learned (EMA) from each block's **measured** wall/nominal, absorbing how fast the machine is right now (heat/load); margin is learned from each request's
# measured underrun (add on a pause, shrink slowly while clean). Neither is a static conservative value -- seamlessness comes from "measured feedback",
# not from inflating constants. The start point mainly uses measured completion times of blocks already arrived; estimates only cover the unsynthesized tail.
# ★2026-07-24: calibrated separately per backend. The original two constants were chatterbox measurements,
# not recalibrated when switching to Qwen3-TTS on 07-22 -- and Qwen3's marginal cost is 2.5x chatterbox's.
# Local measurement (single block, warm model, direct localhost, 2/6/11/15 chars, twice each), linear regression:
#     Qwen3-TTS-12Hz-0.6B-bf16:  wall ≈ 2.545 + 2.707 * audio   (RTF 3.8~12)
# The old SLOPE 1.10 underestimated by 2.5x; speed (a scalar multiplier) cannot correct a slope error, so it jumps with block length:
# audio≈1.0s blocks -> actual/nominal = 5.25/3.60 = 1.46; audio≈2.5s -> 9.31/5.25 = 1.77
# -- exactly why speed oscillated between 1.46 and 1.74 in tts-zh-server.log. Once calibrated, speed
# should converge to ≈1.0 and hold, the start point becomes accurate, and "drops after the first block" (BrokenPipe) decreases.
# Rollback: override with env TTS_FLOOR_NOM / TTS_SLOPE_NOM, or restore the whole file from .bak-preslope-20260724.
if BACKEND == "qwen3":
    _FLOOR_DEFAULT, _SLOPE_DEFAULT = "2.55", "2.71"
else:
    _FLOOR_DEFAULT, _SLOPE_DEFAULT = "2.5", "1.10"   # chatterbox (8123 English) original values, untouched
FLOOR_NOM = float(os.environ.get("TTS_FLOOR_NOM", _FLOOR_DEFAULT))   # nominal fixed overhead, seconds/block
SLOPE_NOM = float(os.environ.get("TTS_SLOPE_NOM", _SLOPE_DEFAULT))   # nominal marginal: synthesis seconds per audio second
_adapt = {"speed": 1.0, "margin": SAFETY_S, "n": 0}
_adapt_lock = threading.Lock()


def _nominal_synth(audio_sec: float) -> float:
    return FLOOR_NOM + SLOPE_NOM * audio_sec


def est_synth_sec(audio_sec: float) -> float:
    with _adapt_lock:
        return _adapt["speed"] * _nominal_synth(audio_sec)


def adapt_margin() -> float:
    with _adapt_lock:
        return _adapt["margin"]


def observe_chunk(audio_sec: float, wall: float):
    """After each block is synthesized, feed back speed (EMA) -- learning how fast the machine currently is relative to the nominal model."""
    if audio_sec < 0.3 or wall <= 0:
        return
    with _adapt_lock:
        obs = wall / _nominal_synth(audio_sec)
        _adapt["speed"] = 0.7 * _adapt["speed"] + 0.3 * obs
        _adapt["n"] += 1


def observe_request(silence_filled: float):
    """At the end of each request, feed back margin: a pause = estimate too shallow -> add; consistently clean -> shrink slowly (approaching minimal seamless latency)."""
    with _adapt_lock:
        if silence_filled > 0.4:
            _adapt["margin"] = min(2.5, _adapt["margin"] + 0.4)
        elif silence_filled < 0.1:
            _adapt["margin"] = max(MARGIN_FLOOR_S, _adapt["margin"] - 0.1)  # never below the floor, to stay seamless


def compute_safe_start(chunks: list, arrivals: list) -> float:
    """Gap-free start point (relative to t0): arrived blocks use **measured** completion times; blocks not yet arrived are extended with the adaptive estimate.
    Recomputed as each block arrives; the closer to start, the less estimation and the more accurate. arrivals=[(arrival time, audio seconds), ...] in block order."""
    n = len(arrivals)
    durs = [d for _, d in arrivals] + [est_audio_sec(chunks[j]) for j in range(n, len(chunks))]
    readies = [a for a, _ in arrivals]
    r = arrivals[-1][0] if arrivals else 0.0
    for j in range(n, len(chunks)):
        r += est_synth_sec(durs[j])            # not-yet-arrived block: estimate serially after the previous one
        readies.append(r)
    safe = cum = 0.0
    for i in range(len(chunks)):
        safe = max(safe, readies[i] - cum)     # block i not late -> start >= ready_i - Σdur_before
        cum += durs[i]
    return safe + adapt_margin()


# ---- Streaming WAV header (data-size=0xFFFFFFFF, play until EOF) ---------------------------
def streaming_wav_header() -> bytes:
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write((0xFFFFFFFF).to_bytes(4, "little"))   # ChunkSize: unknown -> max
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write((16).to_bytes(4, "little"))           # PCM fmt chunk size
    buf.write((1).to_bytes(2, "little"))            # AudioFormat = PCM
    buf.write((1).to_bytes(2, "little"))            # channels = 1
    buf.write(SAMPLE_RATE.to_bytes(4, "little"))
    buf.write(BYTES_PER_SEC.to_bytes(4, "little"))  # byte rate
    buf.write((2).to_bytes(2, "little"))            # block align
    buf.write((16).to_bytes(2, "little"))           # bits per sample
    buf.write(b"data")
    buf.write((0xFFFFFFFF).to_bytes(4, "little"))   # data size: unknown -> max
    return buf.getvalue()


def float_to_pcm(samples: np.ndarray) -> bytes:
    samples = time_stretch(np.asarray(samples, dtype=np.float32).reshape(-1), SPEED_RATE)
    pcm = np.clip(np.asarray(samples, dtype=np.float32) * OUTPUT_GAIN, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


def trim_model_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -25.0,
    padding_seconds: float = 0.04,
) -> np.ndarray:
    """Trim the leading/trailing blank audio Qwen3 adds to every generation, keeping a little natural edge."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    frame_size = max(1, round(sample_rate * 0.01))
    frame_count = (samples.size + frame_size - 1) // frame_size
    padded = np.pad(samples, (0, frame_count * frame_size - samples.size))
    rms = np.sqrt(np.mean(padded.reshape(frame_count, frame_size) ** 2, axis=1) + 1e-12)
    peak = float(rms.max())
    if peak <= 0:
        return samples
    active = np.flatnonzero(rms >= peak * (10.0 ** (threshold_db / 20.0)))
    if active.size == 0:
        return samples
    padding = round(padding_seconds * sample_rate)
    start = max(0, int(active[0]) * frame_size - padding)
    end = min(samples.size, (int(active[-1]) + 1) * frame_size + padding)
    return samples[start:end]


def inter_chunk_pause(chunks: list) -> float:
    """Inter-segment pause: scaled by the estimated **whole-sentence** duration -- not a fixed value, and not by segment length.

    2026-07-25: originally fixed at 0.2s. In a 3.6s long sentence it landed on a comma as a natural break (verified
    "smooth and coherent"), but in a 0.78s short sentence like "Zai ne, ni hao." ("I'm here, hello.") 200ms is 26%, splitting one sentence
    in two -- that is what the user meant by "word by word, never forming a complete sentence". Yet cutting straight to 0.06s
    made long sentences feel "a bit rushed".

    🔴 Do not change this to "scale by segment length" (I tried; the data rejected it): "Qing tian," ("Sunny,", 0.40s) and "Zai ne," ("I'm here,", 0.42s)
    have almost the same segment length, but the user wants 60ms in the short sentence and 200ms at that position in the long one. What decides the pause
    is **how long the whole sentence is**, not how long this segment is -- a 0.8-second utterance should not have a real comma pause at all.

    The ratio 0.06 lands on both points the user approved:
      "Zai ne, ni hao."      d_est 0.84s -> 50ms -> clamped to floor 60ms  (picked by the user from a four-option A/B)
      weather long sentence  d_est 3.50s -> 210ms -> clamped to cap 200ms  (the feel the user had already approved)
    """
    if INTER_CHUNK_PAUSE_S <= 0:
        return 0.0
    total = est_audio_sec(" ".join(chunks))
    return min(INTER_CHUNK_PAUSE_S, max(PAUSE_FLOOR_S, total * PAUSE_RATIO))


def time_stretch(x: np.ndarray, rate: float) -> np.ndarray:
    """WSOLA time stretch without pitch change. rate>1 = faster speech, shorter output.

    ★ 2026-07-25: after listening to B (whole-sentence synthesis) the user said "fine, but a bit slow", and chose 1.1x.
    🔴 Do not go back to model.generate(speed=...) -- see the QWEN_MERGE comment; that parameter is
       not implemented at all in mlx_audio's qwen3 backend (its own docstring says not directly supported yet).
    Why not simple resampling: it raises the pitch too. This is a **cloned voice**; move the pitch and it is no longer the same voice.
    WSOLA = lay frames at a fixed synthesis hop; for each frame, search within ±search in the input for the position with the highest cross-correlation
    with "what should follow the previous frame", then overlap-add -> continuous pitch periods -> faster without pitch change.
    Measured median F0: 1.00x 125.7Hz / 1.20x 121.8Hz / 1.50x 124.7Hz (simple 1.5x resampling would give 188Hz).

    ⚠️ Honest accounting: this happens **after synthesis**, so it saves **not a single second** of first-sound latency; what it saves is the total
       time to finish a sentence. It treats perceived drag, not latency.
    """
    if abs(rate - 1.0) < 1e-3 or x.size == 0:
        return x
    frame, syn_hop, search = 1024, 256, 160
    win = np.hanning(frame).astype(np.float32)
    ana_hop = syn_hop * rate
    out_len = int(len(x) / rate) + frame
    out = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    if len(x) < frame + search:
        return x
    nxt = x[:frame].copy()
    out[:frame] += nxt * win
    wsum[:frame] += win
    y, a = syn_hop, ana_hop
    while y + frame < out_len and int(a) + frame + search < len(x):
        lo = max(0, int(a) - search)
        hi = min(len(x) - frame, int(a) + search)
        if hi <= lo:
            break
        cand = np.lib.stride_tricks.sliding_window_view(x[lo:hi + frame], frame)
        best = lo + int(np.argmax(cand[: hi - lo + 1] @ nxt))
        out[y:y + frame] += x[best:best + frame] * win
        wsum[y:y + frame] += win
        nxt = x[best + syn_hop: best + syn_hop + frame]
        if len(nxt) < frame:
            break
        y += syn_hop
        a += ana_hop
    out, wsum = out[: y + frame], wsum[: y + frame]
    return np.where(wsum > 1e-6, out / np.maximum(wsum, 1e-6), out)


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * seconds)


# =============================================================================
#  Synthesis backend -- single-threaded worker (MLX thread binding); handles all blocks of one utterance at a time
# =============================================================================
_SIM_DIR = os.environ.get("TTS_SIM_DIR")

_jobs: "queue.Queue" = queue.Queue()
_ready = threading.Event()
_load_error: list = []


def _real_synth_setup():
    from mlx_audio.tts.generate import load_audio
    from mlx_audio.tts.utils import load_model
    model = load_model(MODEL_ID)
    ref = load_audio(REF_AUDIO, sample_rate=model.sample_rate)

    if BACKEND == "qwen3":
        if not REF_TEXT:
            raise ValueError("TTS_REF_TEXT is required for the Qwen3 clone backend")
        if not QWEN_TEMPERATURES:
            raise ValueError("TTS_QWEN_TEMPERATURES must contain at least one value")

        def synth(text: str) -> bytes:
            token_limit = max(48, min(144, len(text) * 6))
            for temperature in QWEN_TEMPERATURES:
                results = list(
                    model.generate(
                        text,
                        ref_audio=ref,
                        ref_text=REF_TEXT,
                        lang_code="chinese",
                        temperature=temperature,
                        max_tokens=token_limit,
                        verbose=False,
                    )
                )
                if results and results[-1].token_count < token_limit:
                    samples = trim_model_silence(results[-1].audio, model.sample_rate)
                    return float_to_pcm(samples)
                del results
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except Exception:
                    pass
            raise RuntimeError(f"Qwen3 generation degenerated for {text!r}")

        return synth

    def synth(text: str) -> bytes:
        chunks = []
        for result in model.generate(text, ref_audio=ref, verbose=False, **RECIPE):
            chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        if not chunks:
            return b""
        samples = np.concatenate(chunks)
        samples = filter_profile(samples, model.sample_rate, CLARITY_PROFILE)
        return float_to_pcm(samples)

    return synth


def _sim_synth_setup():
    """Replay backend: read pre-synthesized PCM, sleep for the measured synthesis time, then return. Replaces only 'audio output'; timing is real."""
    def synth(text: str) -> bytes:
        h = hashlib.sha1(text.encode()).hexdigest()
        path = os.path.join(_SIM_DIR, f"{h}.pcm")
        if os.path.exists(path):
            with open(path, "rb") as f:
                pcm = f.read()
            audio_s = len(pcm) / BYTES_PER_SEC
        else:                                   # sentence with no pre-synthesized audio: fall back to silence of the estimated duration
            audio_s = est_audio_sec(text)
            pcm = silence(audio_s)
        synth_wall = max(SIM_FLOOR_S, RTF * audio_s)
        time.sleep(synth_wall)
        return pcm
    return synth


def _generation_thread():
    synth = None

    def _load():
        nonlocal synth
        t0 = time.time()
        synth = _sim_synth_setup() if _SIM_DIR else _real_synth_setup()
        print(f"[{LOG_TAG}] backend loaded in {time.time() - t0:.1f}s "
              f"({'SIM:' + _SIM_DIR if _SIM_DIR else MODEL_ID}); clarity={CLARITY_PROFILE}")

    def _unload():
        """Idle unload -- must happen on this thread (MLX GPU stream thread binding, same as constraint 4).

        ★ del is nowhere near enough: MLX holds GPU memory in its own cache pool.
          Measured 2026-07-17: after a real synthesis 3691MB -> still 3472MB after del+gc -> 668MB after clear_cache().
          With only del, this feature would "look like it runs while saving not a single byte".
        """
        nonlocal synth
        if synth is None:
            return
        synth = None                      # break the closure -> model/ref lose their last reference
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()              # ★ this line is what actually releases GPU memory, not del
        except Exception as exc:
            print(f"[{LOG_TAG}] ! clear_cache failed (non-fatal, continuing to serve): {exc!r}")
        print(f"[{LOG_TAG}] ~ idle {IDLE_UNLOAD_S}s -> unloaded, ~3.0GB GPU memory released; "
              f"next request reloads automatically (first sentence +~2.9s)")

    try:
        _load()                           # load at startup: keeps the 07-16 _ready semantics unchanged
    except Exception as exc:
        _load_error.append(repr(exc))
        _ready.set()
        return
    _ready.set()

    while True:
        try:
            job = _jobs.get(timeout=IDLE_UNLOAD_S) if IDLE_UNLOAD_S > 0 else _jobs.get()
        except queue.Empty:
            _unload()
            job = _jobs.get()             # after unload, wait indefinitely until real work arrives
        chunks, out = job
        if chunks is None:
            return
        try:
            if synth is None:             # was unloaded -> reload
                _load()
            _pause = inter_chunk_pause(chunks)   # one value per sentence, computed once outside the loop
            for chunk_index, ct in enumerate(chunks):
                # ★ w0 is taken after _load(): the ~2.0s reload must never count toward wall,
                #   or observe_chunk() would feed it back into speed as "machine got slower", poisoning the estimator.
                #   The extra 0.8s the device waits is absorbed naturally by the measured arrival times in arrivals (the start point shifts later).
                w0 = time.time()
                pcm = synth(ct)
                if _pause > 0 and chunk_index + 1 < len(chunks):
                    pcm += silence(_pause)
                out.put(("pcm", pcm, time.time() - w0))   # include this block's measured synthesis wall, for feedback
            out.put(("done", None, 0.0))
        except Exception as exc:
            out.put(("error", repr(exc), 0.0))


# =============================================================================
#  Streaming response: delayed start + silence fill on underrun + close-delimited framing
# =============================================================================
def stream_utterance(sock: socket.socket, text: str, log_prefix="") -> dict:
    """Synthesize text and stream it into sock. Returns timing stats. Sends no Content-Length; closes the connection at the end."""
    raw = text.strip()
    clean = sanitize(raw)                        # sanitize first (em dash and other non-ASCII folded/dropped, to protect the voice)
    stats = {"clean": clean}
    if clean != raw:                             # leave a trace: say so when sanitizing changed anything (same as production tts_server.py)
        print(f"[{LOG_TAG}] ~ sanitize {raw!r} -> {clean!r}")
    if not clean:
        sock.sendall(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
        return stats

    if BACKEND == "qwen3":
        sentences = split_qwen_segments(clean)   # Qwen3: comma-level short segments, to avoid degradation on long segments
        # ★ 2026-07-25: TTS_QWEN_MERGE -- **off by default; when off, behavior is byte-for-byte unchanged**.
        # Why this switch exists: measured cost per synthesis ≈ 2.5s fixed + 2.7 x audio seconds, and **the fixed part is charged
        # per block**. Comma-level splitting turns an 18-character reply into 2-4 blocks -> 5-10s of fixed cost alone,
        # the bulk of first-sound latency, and nearly independent of how long the user's message is. Merging into one block saves 2.5s x (blocks-1).
        # The cost is the "avoid degradation on long segments" noted above -- voice quality/phrasing may get worse, **which only the user's ears can judge**,
        # so it is a switch, off by default, decided by A/B.
        if QWEN_MERGE:
            chunks = [clean]                     # synthesize the whole sentence at once; commas stay in the text for the model to phrase
        else:
            chunks = sentences
    else:
        sentences = split_sentences(clean)       # Chatterbox: original sentence-level blocking strategy
        chunks = make_chunks(sentences)
    stats.update(sentences=len(sentences), chunks=len(chunks),
                 d_est=round(est_audio_sec(clean), 2), policy=START_POLICY)
    stats["segs"] = list(chunks)          # ★ 2026-07-25: for diagnosing "silence within a segment"
    _seg_durs = []                        # actual audio seconds produced per segment, in arrival order

    out: "queue.Queue" = queue.Queue()
    t0 = time.time()
    _jobs.put((chunks, out))                      # hand off to the single-threaded synthesis worker

    buffered = bytearray()                        # PCM synthesized but not yet released
    buffered_audio = 0.0
    done = False
    started = False
    header = streaming_wav_header()

    def begin_stream():
        sock.sendall(b"HTTP/1.1 200 OK\r\n"
                     b"Content-Type: audio/wav\r\n"
                     b"Connection: close\r\n\r\n")
        sock.sendall(header)

    # ---- Phase A: accumulate headstart (send no bytes meanwhile; the device waits on POST for response headers) ----
    # smooth: the start point is recomputed by compute_safe_start as each block arrives (arrived blocks use measured completion times);
    # early: start as soon as the first block is in hand. Real underruns are backstopped by phase-B silence fill, never cut off.
    arrivals = []                                 # [(arrival time rel t0, audio seconds), ...] in block order
    while not started:
        ss = 0.0 if START_POLICY == "early" else compute_safe_start(chunks, arrivals)
        timeout = max(0.0, (t0 + ss) - time.time())
        try:
            kind, payload, wall = out.get(timeout=timeout if timeout > 0 else 0.02)
        except queue.Empty:
            kind = None
        if kind == "error":
            sock.sendall(b"HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n")
            stats["error"] = payload
            return stats
        if kind == "done":
            done = True
        elif kind == "pcm":
            dur = len(payload) / BYTES_PER_SEC
            _seg_durs.append(round(dur, 3))
            arrivals.append((round(time.time() - t0, 3), dur))
            observe_chunk(dur, wall)              # feed back measured speed
            buffered += payload
            buffered_audio = len(buffered) / BYTES_PER_SEC
        ss = 0.0 if START_POLICY == "early" else compute_safe_start(chunks, arrivals)
        if buffered and (time.time() - t0 >= ss or done):
            started = True
            stats["t_first_audio"] = round(time.time() - t0, 2)
            stats["headstart_audio"] = round(buffered_audio, 2)
            begin_stream()
            sock.sendall(bytes(buffered))
            buffered = bytearray()

    # ---- Phase B: keep feeding after playback starts ----
    # Silence is filled by the **server's own real-time clock** (not whenever the queue is empty): estimated device buffer = audio sent - time played;
    # silence is filled only when that is below LOW_WATER and the real block has not arrived. So silence is never inserted early between chunk0 and chunk1,
    # and it does not depend on TCP backpressure -- loopback tests reflect the real device experience too. Silence on a real underrun = a short degraded pause, never a cutoff.
    silence_filled = 0.0
    t_play0 = time.time()                         # playback start wall clock (the device consumes in real time from about now)
    written_audio = buffered_audio                # audio seconds already sent in phase A
    while not done:
        ahead = written_audio - (time.time() - t_play0)   # server's estimate of the device buffer headroom (s)
        try:
            kind, payload, wall = out.get(timeout=0.05)
        except queue.Empty:
            if ahead < LOW_WATER:                 # device nearly out of data and the real block has not arrived -> fill a frame of silence to survive
                sock.sendall(silence(SILENCE_FRAME))
                written_audio += SILENCE_FRAME
                silence_filled += SILENCE_FRAME
            continue                              # enough headroom: just wait for the real block, inject no silence
        if kind == "error":
            stats["error"] = payload
            break
        if kind == "done":
            done = True
            break
        observe_chunk(len(payload) / BYTES_PER_SEC, wall)   # later blocks also feed back measured speed
        _seg_durs.append(round(len(payload) / BYTES_PER_SEC, 3))
        sock.sendall(payload)                     # TCP backpressure = the device's real-time consumption naturally limits the rate
        written_audio += len(payload) / BYTES_PER_SEC

    stats["seg_durs"] = _seg_durs
    stats["t_done"] = round(time.time() - t0, 2)
    stats["silence_filled"] = round(silence_filled, 2)
    observe_request(silence_filled)              # feed back measured margin
    with _adapt_lock:
        stats["adapt"] = {"speed": round(_adapt["speed"], 2), "margin": round(_adapt["margin"], 2)}
    try:
        sock.shutdown(socket.SHUT_WR)             # half-close: tell the device the stream has ended (EOF)
    except OSError:
        pass
    return stats


# ---- Minimal HTTP parsing (only method/path/body needed) --------------------------------
class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        sock.settimeout(70)
        try:
            data = self._read_headers(sock)
        except (OSError, ValueError):
            return
        if data is None:
            return
        head, rest = data
        line = head.split("\r\n", 1)[0]
        try:
            method, path, _ = line.split(" ", 2)
        except ValueError:
            return

        if path.startswith("/health"):
            self._health(sock)
            return
        if method != "POST" or not path.startswith("/v1/audio/speech"):
            sock.sendall(b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n")
            return

        clen = 0
        for h in head.split("\r\n")[1:]:
            if h.lower().startswith("content-length:"):
                clen = int(h.split(":", 1)[1].strip())
        body = rest
        while len(body) < clen:
            more = sock.recv(65536)
            if not more:
                break
            body += more
        try:
            text = json.loads(body.decode("utf-8", "replace")).get("input", "")
        except (ValueError, AttributeError):
            sock.sendall(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
            return

        if not _ready.is_set():
            sock.sendall(b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n")
            return
        if _load_error:
            sock.sendall(b"HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n")
            return

        stats = stream_utterance(sock, text)
        print(f"[{LOG_TAG}] {stats}")

    def _read_headers(self, sock):
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                return None
            buf += chunk
            if len(buf) > 1 << 20:
                raise ValueError("header too large")
        head, rest = buf.split(b"\r\n\r\n", 1)
        return head.decode("latin1"), rest

    def _health(self, sock):
        if not _ready.is_set():
            body = b'{"status":"loading"}'
            code = b"503 Service Unavailable"
        elif _load_error:
            body = json.dumps({"status": "error", "detail": _load_error[0]}).encode()
            code = b"500 Internal Server Error"
        else:
            body = json.dumps({"status": "ok", "backend": _SIM_DIR or MODEL_ID,
                               "port": PORT, "clarity": CLARITY_PROFILE,
                               "lang_code": LANG_CODE}).encode()
            code = b"200 OK"
        sock.sendall(b"HTTP/1.1 " + code + b"\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() +
                     b"\r\nConnection: close\r\n\r\n" + body)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    threading.Thread(target=_generation_thread, daemon=True).start()
    print(f"[{LOG_TAG}] loading backend ...")
    _ready.wait()
    if _load_error:
        raise SystemExit(f"[{LOG_TAG}] backend load failed: {_load_error[0]}")
    with Server((HOST, PORT), Handler) as srv:
        print(f"[{LOG_TAG}] listening on {HOST}:{PORT}")
        srv.serve_forever()
