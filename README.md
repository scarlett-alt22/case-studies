# Case studies

Engineering case studies from a one-person setup of local AI services. Each folder is self-contained: the write-up is its `README.md`, and the raw data and scripts it cites by file name and line number sit beside it, so the numbers can be recomputed without trusting the author.

| # | Case study | Topic |
|---|---|---|
| 01 | [When the Model Is Slower Than Playback, the Stream Cannot Pause](./01-streaming-tts/) | Continuity-first streaming TTS for a failure-intolerant audio client, and the audit that overturned three of its findings |

## Changes to the source snapshot

`01-streaming-tts/tts_stream_server.py` is a 2026-07-25 source snapshot. Before publishing:

- one local file path was replaced with `/path/to/…`, a personal name was removed from seven comments, and project-specific names for the device and the voice were replaced with neutral descriptions;
- comments, docstrings and three log messages were translated from Chinese to English;
- the two sentence-splitting regexes now write their full-width punctuation as `\uXXXX` escapes instead of literal characters (same characters, same matches).

No code was otherwise changed, and no lines were added or removed, so every line number cited in the write-up still points at the same code.

## License

Code (`*.py`) is released under the [MIT License](./LICENSE). Write-ups and measurement data (`*.md`, `*.jsonl`) are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
