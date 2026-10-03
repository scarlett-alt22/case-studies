#!/usr/bin/env python3
"""Randomized single-block timing probe for case study 01.

The live endpoint does not expose the worker's private per-chunk ``wall`` value.
For a one-block request, body TTFB is the closest non-invasive proxy: the server
does not send response headers or PCM until that block is complete. The proxy can
still include queue wait or an idle model reload, so both are retained as explicit
limitations rather than subtracted or inferred away.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import json
import random
import struct
import sys
import time
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8123
SPEECH_PATH = "/v1/audio/speech"
HEALTH_PATH = "/health"
OUTPUT = Path("codex-single-block-measurements-2026-08-06.jsonl")
SEED = 20260806
FOLLOWUP_SEED = 42
GAP_S = 3

# No sentence-ending punctuation: the snapshot's splitter produces exactly one
# sentence and make_chunks() therefore produces exactly one synthesis block.
TEXTS = [
    {"text_id": "short_a", "text": "Calm voices help"},
    {"text_id": "short_b", "text": "Clear speech works"},
    {"text_id": "medium_a", "text": "Steady voices make timing easier"},
    {"text_id": "medium_b", "text": "Careful tests reveal hidden delays"},
    {"text_id": "long_a", "text": "A steady voice makes repeated timing easier to compare"},
    {"text_id": "long_b", "text": "Careful local tests separate synthesis time from playback time"},
    {"text_id": "xlong_a", "text": "A measured voice helps engineers compare repeated synthesis runs without guessing"},
    {"text_id": "xlong_b", "text": "Randomized local trials make warmup and changing machine load easier to notice"},
    {"text_id": "xxlong_a", "text": "A carefully measured voice helps engineers compare repeated local synthesis runs while preserving every observed result"},
    {"text_id": "xxlong_b", "text": "Randomized single block trials make warmup queue delay and changing machine load easier for another engineer to audit"},
]
WARMUP = {"text_id": "warmup", "text": "Warm up the existing speech model"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def health() -> dict[str, object]:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=10)
    started = time.perf_counter_ns()
    conn.request("GET", HEALTH_PATH)
    response = conn.getresponse()
    body = response.read()
    elapsed = (time.perf_counter_ns() - started) / 1e9
    headers = dict(response.getheaders())
    status = response.status
    conn.close()
    try:
        parsed: object = json.loads(body)
    except ValueError:
        parsed = body.decode("utf-8", "replace")
    return {
        "record_type": "health",
        "at_utc": utc_now(),
        "status": status,
        "elapsed_s": round(elapsed, 6),
        "headers": headers,
        "body": parsed,
        "interpretation": "worker ready flag only; does not prove model is resident",
    }


def parse_wav_header(data: bytes) -> dict[str, object]:
    if len(data) < 44:
        return {"header_error": f"only {len(data)} bytes"}
    return {
        "riff": data[0:4].decode("ascii", errors="replace"),
        "riff_size_field": struct.unpack_from("<I", data, 4)[0],
        "wave": data[8:12].decode("ascii", errors="replace"),
        "fmt": data[12:16].decode("ascii", errors="replace"),
        "fmt_size": struct.unpack_from("<I", data, 16)[0],
        "audio_format": struct.unpack_from("<H", data, 20)[0],
        "channels": struct.unpack_from("<H", data, 22)[0],
        "sample_rate": struct.unpack_from("<I", data, 24)[0],
        "byte_rate": struct.unpack_from("<I", data, 28)[0],
        "block_align": struct.unpack_from("<H", data, 32)[0],
        "bits_per_sample": struct.unpack_from("<H", data, 34)[0],
        "data": data[36:40].decode("ascii", errors="replace"),
        "data_size_field": struct.unpack_from("<I", data, 40)[0],
    }


def run_one(run_index: int, item: dict[str, str], phase: str) -> dict[str, object]:
    text = item["text"]
    payload = json.dumps({"input": text}).encode("utf-8")
    conn = http.client.HTTPConnection(HOST, PORT, timeout=120)
    start_utc = utc_now()
    start_ns = time.perf_counter_ns()
    conn.request(
        "POST",
        SPEECH_PATH,
        body=payload,
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
    )
    response = conn.getresponse()
    headers_ns = time.perf_counter_ns()
    reads: list[dict[str, object]] = []
    body_parts: list[bytes] = []
    while True:
        chunk = response.read1(65536)
        arrived_ns = time.perf_counter_ns()
        if not chunk:
            break
        body_parts.append(chunk)
        reads.append({"elapsed_s": round((arrived_ns - start_ns) / 1e9, 6), "bytes": len(chunk)})
    end_ns = time.perf_counter_ns()
    headers = dict(response.getheaders())
    status = response.status
    conn.close()

    body = b"".join(body_parts)
    wav = parse_wav_header(body[:44])
    sample_rate = int(wav.get("sample_rate", 24000))
    channels = int(wav.get("channels", 1))
    bits_per_sample = int(wav.get("bits_per_sample", 16))
    bytes_per_second = sample_rate * channels * bits_per_sample / 8
    audio_s = (len(body) - 44) / bytes_per_second if len(body) >= 44 else None
    ttfb = reads[0]["elapsed_s"] if reads else None
    total = round((end_ns - start_ns) / 1e9, 6)
    return {
        "record_type": "run",
        "phase": phase,
        "run_index": run_index,
        "start_utc": start_utc,
        "text_id": item["text_id"],
        "input": text,
        "input_chars": len(text),
        "input_utf8_bytes": len(text.encode("utf-8")),
        "expected_sentences": 1,
        "expected_chunks": 1,
        "status": status,
        "response_headers": headers,
        "headers_s": round((headers_ns - start_ns) / 1e9, 6),
        "block_wall_proxy_s": ttfb,
        "proxy_definition": "request start to first de-framed body byte for a one-block request",
        "proxy_includes": ["possible queue wait", "possible idle reload", "HTTP/loopback overhead"],
        "ttfb_s": ttfb,
        "total_s": total,
        "tail_after_ttfb_s": round(total - float(ttfb), 6) if ttfb is not None else None,
        "body_bytes": len(body),
        "generated_pcm_s_from_bytes": round(audio_s, 6) if audio_s is not None else None,
        "pcm_duration_note": "one block has no scheduler fill silence, but may retain model-produced edge silence",
        "sha256": hashlib.sha256(body).hexdigest(),
        "wav_header": wav,
        "reads": reads,
    }


def append(record: dict[str, object]) -> None:
    with OUTPUT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--followup":
        followup_pool = [item for item in TEXTS if item["text_id"] in {
            "short_a", "medium_a", "long_a", "xlong_a", "xxlong_a"
        }]
        rng = random.Random(FOLLOWUP_SEED)
        rng.shuffle(followup_pool)
        append({
            "record_type": "followup_metadata",
            "created_utc": utc_now(),
            "reason": "exact repeats in a second seeded order to reduce length/time-drift confounding",
            "random_seed": FOLLOWUP_SEED,
            "gap_s": GAP_S,
            "order": [item["text_id"] for item in followup_pool],
        })
        append(health())
        for run_index, item in enumerate(followup_pool, start=11):
            result = run_one(run_index, item, "measured_followup")
            append(result)
            print(
                f"run={run_index} id={item['text_id']} chars={result['input_chars']} "
                f"proxy={result['block_wall_proxy_s']:.3f}s "
                f"audio={result['generated_pcm_s_from_bytes']:.3f}s "
                f"tail={result['tail_after_ttfb_s']:.3f}s",
                flush=True,
            )
            if run_index != 10 + len(followup_pool):
                time.sleep(GAP_S)
        append(health())
        return

    rng = random.Random(SEED)
    order = TEXTS.copy()
    rng.shuffle(order)
    metadata = {
        "record_type": "metadata",
        "created_utc": utc_now(),
        "host": HOST,
        "port": PORT,
        "path": SPEECH_PATH,
        "clock": "time.perf_counter_ns",
        "read_method": "http.client.HTTPResponse.read1(65536), de-framed body",
        "single_block_basis": "inputs contain no sentence-ending punctuation; snapshot split_sentences/make_chunks yields one chunk",
        "duration_formula": "(body_bytes - 44) / (sample_rate * channels * bits_per_sample/8)",
        "random_seed": SEED,
        "gap_s": GAP_S,
        "order": [item["text_id"] for item in order],
        "important_limit": "block_wall_proxy_s is not the unexposed worker wall and can include queue/reload time",
    }
    OUTPUT.write_text(json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8")
    append(health())

    warmup = run_one(0, WARMUP, "warmup")
    append(warmup)
    print(
        f"phase=warmup proxy={warmup['block_wall_proxy_s']:.3f}s "
        f"audio={warmup['generated_pcm_s_from_bytes']:.3f}s",
        flush=True,
    )
    time.sleep(GAP_S)

    for run_index, item in enumerate(order, start=1):
        result = run_one(run_index, item, "measured")
        append(result)
        print(
            f"run={run_index} id={item['text_id']} chars={result['input_chars']} "
            f"proxy={result['block_wall_proxy_s']:.3f}s "
            f"audio={result['generated_pcm_s_from_bytes']:.3f}s "
            f"tail={result['tail_after_ttfb_s']:.3f}s",
            flush=True,
        )
        if run_index != len(order):
            time.sleep(GAP_S)

    append(health())


if __name__ == "__main__":
    main()
