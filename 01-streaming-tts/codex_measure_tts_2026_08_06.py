#!/usr/bin/env python3
"""Independent loopback timing measurement for case study 01.

Records raw per-read timing metadata but deliberately does not retain generated WAVs.
Uses only Python's standard library.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import json
import struct
import time
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8123
PATH = "/v1/audio/speech"
OUTPUT = Path("codex-measurements-case-study-01-2026-08-06.jsonl")
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "A steady stream of speech helps us measure when the first audio arrives.",
    "Clear timing data makes it easier to separate synthesis cost from playback behavior.",
]

# A balanced order limits a simple warm-up/drift effect from being confounded with length.
ORDER = [1, 2, 3, 3, 1, 2, 2, 3, 1]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_canonical_wav_header(data: bytes) -> dict[str, object]:
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


def run_one(run_index: int, sentence_count: int) -> dict[str, object]:
    text = " ".join(SENTENCES[:sentence_count])
    payload = json.dumps({"input": text}).encode("utf-8")
    conn = http.client.HTTPConnection(HOST, PORT, timeout=120)
    start_wall = utc_now()
    start_ns = time.perf_counter_ns()
    conn.request(
        "POST",
        PATH,
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
        reads.append(
            {
                "elapsed_s": round((arrived_ns - start_ns) / 1e9, 6),
                "bytes": len(chunk),
            }
        )
    end_ns = time.perf_counter_ns()
    conn.close()

    body = b"".join(body_parts)
    wav = parse_canonical_wav_header(body[:44])
    sample_rate = int(wav.get("sample_rate", 24000))
    channels = int(wav.get("channels", 1))
    bits_per_sample = int(wav.get("bits_per_sample", 16))
    bytes_per_second = sample_rate * channels * bits_per_sample / 8
    audio_s = (len(body) - 44) / bytes_per_second if len(body) >= 44 else None
    return {
        "run_index": run_index,
        "start_utc": start_wall,
        "sentence_count": sentence_count,
        "input": text,
        "input_utf8_bytes": len(text.encode("utf-8")),
        "status": response.status,
        "response_headers": dict(response.getheaders()),
        "headers_s": round((headers_ns - start_ns) / 1e9, 6),
        "ttfb_s": reads[0]["elapsed_s"] if reads else None,
        "total_s": round((end_ns - start_ns) / 1e9, 6),
        "body_bytes": len(body),
        "audio_s_from_pcm_bytes": round(audio_s, 6) if audio_s is not None else None,
        "sha256": hashlib.sha256(body).hexdigest(),
        "wav_header": wav,
        "reads": reads,
    }


def main() -> None:
    metadata = {
        "record_type": "metadata",
        "created_utc": utc_now(),
        "host": HOST,
        "port": PORT,
        "path": PATH,
        "clock": "time.perf_counter_ns",
        "read_method": "http.client.HTTPResponse.read1(65536), de-chunked body",
        "duration_formula": "(body_bytes - 44) / (sample_rate * channels * bits_per_sample/8)",
        "sentences": SENTENCES,
        "order": ORDER,
    }
    OUTPUT.write_text(json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8")
    for run_index, sentence_count in enumerate(ORDER, start=1):
        result = run_one(run_index, sentence_count)
        with OUTPUT.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"run={run_index} sentences={sentence_count} "
            f"ttfb={result['ttfb_s']:.3f}s total={result['total_s']:.3f}s "
            f"audio={result['audio_s_from_pcm_bytes']:.3f}s "
            f"bytes={result['body_bytes']} reads={len(result['reads'])}",
            flush=True,
        )
        if run_index != len(ORDER):
            time.sleep(2)


if __name__ == "__main__":
    main()
