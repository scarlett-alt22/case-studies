# When the Model Is Slower Than Playback, the Stream Cannot Pause

*Designing continuity-first streaming TTS for a failure-intolerant audio client—and auditing the evidence after it shipped*

A small ESP32 speaker on my desk answers questions using a locally hosted cloned voice. The model returns complete waveforms rather than useful partial audio, synthesis takes longer than the resulting audio takes to play, and the client can terminate a response if the stream starves after playback begins.

I could not make the model faster. I could decide how to split the text, when to release the first completed block, and how to keep later blocks ahead of playback. That turned a latency problem into a scheduling problem.

The original write-up led with a percentage improvement. I have withdrawn it. Its denominator was a historical reading that the working notes later marked as contaminated by padded silence. Fresh evidence cannot repair a bad baseline. The result worth publishing is the design judgment: when generation trails playback and starvation is terminal, chunk boundaries and start policy are correctness decisions, not tuning details.

## How this revision was audited

The scheduler described here is mine. The audit that produced every fresh number in it is not, and saying so is load-bearing rather than modest.

Before publishing I had two AI coding agents re-derive the claims independently. The first re-measured the endpoint and wrote up its findings. The second was given the same access and asked to review that write-up cold and re-run the measurements itself — explicitly instructed to take its own readings *before* reading the first agent's conclusions, and to state in its report whether it had honoured that order. It did, and it then overturned three of the first agent's findings, including one that had already been written down as a result. Section 5 is about that failure, because it is the most useful thing that happened.

I am naming this arrangement rather than absorbing it into a first person that did everything, for the same reason the numbers below carry their spread: **a claim is worth what its checking procedure is worth.** A finding that survived an adversarial second pass by a party with no stake in the first answer is a different object from a finding I re-read and still agreed with.

## Evidence boundary

This case study keeps three evidence classes separate:

- **Fresh measurement.** For this revision, the second agent measured localhost requests to `POST 127.0.0.1:8123/v1/audio/speech` on 2026-08-06. The two JSONL files preserve every request, input, timestamp, body read, response header, WAV field, byte count, and hash: [`codex-measurements-case-study-01-2026-08-06.jsonl`](./codex-measurements-case-study-01-2026-08-06.jsonl) and [`codex-single-block-measurements-2026-08-06.jsonl`](./codex-single-block-measurements-2026-08-06.jsonl). The measurement scripts are published beside them, so the fit below can be recomputed by a third party without trusting any of us.
- **Snapshot evidence.** [`tts_stream_server.py`](./tts_stream_server.py) is a 2026-07-25 source copy, not proof of the live process. The measured endpoint and response framing match it closely, but environment overrides are not reported by the endpoint and the snapshot defaults differ from the observed service on at least port and clarity profile.
- **Historical record.** Earlier device inspection, working-log timings, simulator runs, and post-ship incidents explain why decisions were made. Unless explicitly marked fresh, they are not current performance claims.

Three distinctions keep unlike numbers from trading places:

1. **Body-TTFB is not device first sound.** The fresh client timer stops at the first de-framed response-body read—sometimes only the 44-byte WAV header. Device parsing, buffering, and playback scheduling still follow.
2. **PCM byte duration is not pure voiced duration.** `(body bytes − 44) / 48,000` is valid for the measured 24 kHz, mono, 16-bit stream, but a multi-block stream may include scheduler-filled silence, and even one block may contain model-produced edge silence.
3. **`/health = ok` is not proof that the model is resident.** In the snapshot, idle unload does not clear the ready flag. A fresh health response was immediately followed by a 12.045 s warm-up request that produced 2.58 s of PCM ([single-block JSONL lines 2–3](./codex-single-block-measurements-2026-08-06.jsonl)). Reload and shared-queue wait are both consistent with that delay; the endpoint cannot distinguish them.

## 1. Constraints: continuity shaped the design

The client was already capable of consuming a real stream. Historical device inspection traced a 30 KB audio buffer feeding WAV playback as bytes arrived. It also found a narrow failure window: about 0.64 s of buffered audio plus a 500 ms blocking read before the client ended the HTTP response. Those device figures were not remeasured for this article, so the often-repeated “about 1.1 s” is a historical engineering constraint, not a fresh device result.

The transport imposed two more constraints.

First, the WAV header needed an unknown length. The server writes `0xFFFFFFFF` into both length fields and lets socket close mark the end. Every fresh response reproduced those placeholder fields while delivering 24 kHz, mono, 16-bit PCM; duration therefore had to come from received bytes, not the header ([request JSONL lines 2–10](./codex-measurements-case-study-01-2026-08-06.jsonl)).

Second, historical inspection found that the device read the raw HTTP stream without decoding chunked transfer framing. The response therefore had to be close-delimited: `Connection: close`, no `Content-Length`, and no chunk markers. Fresh measurement confirmed `Content-Type: audio/wav` and `Connection: close` on every row. The raw-socket server was not a preference; it was the compatibility layer.

The final constraint was computational. In every one of the 15 post-warm-up one-block runs, the client wall proxy exceeded the PCM duration it produced. The ratio had a median of 1.815 and a range of 1.573–2.940. This is not a pure model real-time factor—the proxy can contain queue, reload, and loopback overhead—but it is enough to establish the inequality the delivery path must survive: blocks become available more slowly than their audio is consumed.

Once playback starts, naive “send each sentence when ready” streaming has no continuity guarantee. The first block creates a deadline for every block behind it.

## 2. Three assumptions that failed

### “The model streams, so this is plumbing”

The apparent streaming option did not provide usable partial waveform output for this voice model. The snapshot still shows the effective boundary: a model call collects generated audio, concatenates it, applies post-processing, and returns a complete PCM block (`tts_stream_server.py:498–506`).

I therefore had to manufacture streaming above the model call. Text became multiple calls, and transport plumbing became a scheduling problem.

### “Generation is close enough to real time”

It was not. The fresh one-block rows establish direction without pretending the client proxy is an internal model timer: every measured block took longer to arrive than its output took to play. That makes underrun protection load-bearing rather than defensive polish.

### “Smaller blocks are always cheaper”

A small first block helps because no audio can start before that block exists. Small later blocks hurt because each model call repeats fixed work. Fresh short inputs make the trade visible: 1.94 s, 2.06 s, and 2.14 s of PCM arrived after body-TTFB proxies of 5.635 s, 5.831 s, and 6.291 s ([single-block JSONL lines 5, 9, and 21](./codex-single-block-measurements-2026-08-06.jsonl)).

The first block wants to be small. Later blocks need to be large enough to amortize call overhead. Uniform chunking is simple and wrong at both ends.

### A fresh benchmark at the unit the claim describes

The working log contained a historical estimate for per-call fixed cost. Whole multi-sentence requests cannot test it: request totals can mix several model calls, scheduler delay, filled silence, reload, and queue time.

The reviewing agent reran the benchmark with inputs containing no sentence-ending punctuation, so the snapshot splitter would produce one sentence and one synthesis block. After a separately recorded warm-up, ten inputs ran in seeded random order. Because the first shuffle still happened to put several long inputs early and short ones late, five exact texts were repeated in a second seeded order. Nothing was deleted. The live endpoint does not expose the worker’s private per-block `wall`, so the measured value remains a client proxy from request start to first body byte.

| JSONL line | Run | Input chars | PCM from bytes (s) | One-block wall proxy (s) |
|---:|---:|---:|---:|---:|
| 4 | 1 | 34 | 2.94 | 7.394 |
| 5 | 2 | 18 | 1.94 | 5.635 |
| 6 | 3 | 78 | 5.14 | 9.569 |
| 7 | 4 | 117 | 8.06 | 14.632 |
| 8 | 5 | 119 | 8.78 | 15.397 |
| 9 | 6 | 16 | 2.06 | 5.831 |
| 10 | 7 | 32 | 3.14 | 6.092 |
| 11 | 8 | 81 | 5.90 | 9.504 |
| 12 | 9 | 54 | 4.54 | 7.716 |
| 13 | 10 | 62 | 4.50 | 7.671 |
| 17 | 11 | 81 | 6.34 | 9.974 |
| 18 | 12 | 32 | 3.06 | 5.906 |
| 19 | 13 | 54 | 4.34 | 7.517 |
| 20 | 14 | 119 | 8.26 | 13.508 |
| 21 | 15 | 16 | 2.14 | 6.291 |

The complete rows are in [`codex-single-block-measurements-2026-08-06.jsonl`](./codex-single-block-measurements-2026-08-06.jsonl), and the reproducer is [`codex_measure_tts_single_blocks_2026_08_06.py`](./codex_measure_tts_single_blocks_2026_08_06.py). From those 15 rows:

```text
one-block body-TTFB proxy ≈ 2.368 + 1.365 × output PCM duration
R² = 0.931, n = 15
```

That line is descriptive, not a production constant. Text changes the generated waveform; exact repeats produced different PCM durations; machine load drifted; queueing and reload were not observable; and the dependent variable is outside the worker. The useful conclusion is narrower: repeated call overhead is large enough that fine-grained chunking is expensive, but the historical fixed-cost estimate is not a current measurement.

## 3. Two optimizations I killed before building them

The first tempting optimization was caching voice conditioning because it was recomputed on every call. The working log reported 0.16 s for that step. That microbenchmark was not rerun for this article, and both reviewing agents flagged it as unrevalidated. Even if the historical reading was accurate, it was small next to multi-second block times and did not justify persistent shared state in a single-threaded worker.

The second was padding inputs into shape buckets to avoid recompilation. A historical microbenchmark reported 5.68 s for a first shape and 5.99 s for its repeat. The repeat was not faster, so an invasive padding path—with wasted inference and more complicated chunking—had no measured case. Those timings remain unrevalidated history, not current constants.

Both decisions used the same rule: measure a suspected fixed cost before building an optimization around it. Repeated work and expensive work are not synonyms.

## 4. What I built

If the July snapshot matches the live scheduler, the path is:

```text
sanitize → split sentences → group synthesis blocks → one synthesis worker
         → choose a safe release time → close-delimited WAV → client playback
```

The interesting work sits in four scheduling decisions.

### Asymmetric grouping

The first sentence becomes the first block so audio can become available early. Later sentences accumulate greedily so repeated model calls can be amortized. The snapshot targets an estimated 2.6 s of audio for later blocks (`tts_stream_server.py:232–246`). That is a snapshot configuration, not a freshly calibrated optimum.

### Two honest start policies

`early` releases the first completed block immediately. It minimizes body-TTFB but may need silence while later blocks catch up.

`smooth` waits until the scheduler predicts that every later block can arrive before playback reaches it. It optimizes continuity, not the smallest isolated latency number.

The product judgment is portable: waiting before speech begins can read as thinking; stopping halfway through a sentence reads as failure. “Earliest” and “smoothest” are different product choices, so the code names both instead of hiding the trade-off inside one latency metric.

### An estimator that gets more empirical as release approaches

The snapshot estimates ungenerated blocks with a fixed floor and slope, multiplied by an adaptive speed value. As blocks finish, it updates that speed from per-block wall and audio duration, prefers actual completion times for blocks already available, and estimates only the remaining tail (`tts_stream_server.py:253–323`). A separate margin widens after material filled silence and relaxes after clean requests.

The live endpoint exposes none of those current estimator values. `/health` does not report the active floor, slope, speed, margin, queue wait, or loaded state, so this article makes no fresh convergence claim.

### Underrun protection based on media time

An early implementation reportedly inserted silence whenever the send queue was empty. A fast socket can drain its send queue while the client still has audio buffered, so queue emptiness is not evidence of starvation.

The snapshot instead estimates `audio written − playback elapsed` against the server’s clock and fills silence only when that media-time lead falls below a low-water condition (`tts_stream_server.py:677–708`). That decouples correctness from incidental TCP backpressure.

The simulator follows the same principle. It can replay previously captured PCM on captured synthesis timing against a virtual consumption model, so framing, release timing, and starvation behavior can be tested without loading a second model. A real device is still required to validate acoustic first sound and perceived continuity; localhost body-TTFB cannot substitute for either.

## 5. How it broke after the design was correct

The post-ship failures changed how I trust performance work more than the scheduler itself did.

### A green test missed the risky path

The rewrite moved text sanitization, but the smoke test used pure ASCII. It could pass whether typographic punctuation handling survived or not. An em dash was the smallest input that actually crossed the changed boundary.

The lesson was not “add more tests.” It was to name what the change could have broken, then choose the smallest input that exercises it.

### Successful runs left no evidence

Historical device runs completed and produced audible output, but buffered service logs disappeared when the process ended. The outcome existed; the trace did not. Unbuffered logging became part of the measurement system rather than a debugging preference.

### Loopback hid the failure mechanism

Historical tests initially treated loopback socket behavior as if it were device backpressure. Local buffering absorbed the stalls under test. The clock-based pacing rule repaired both the scheduler and its testability: media time, not queue or socket state, became the invariant.

### Two flattering numbers were caught inside the project

One historical latency baseline came from a run the notes later marked as padded with roughly 19 s of silence. Another underrun figure came from a reader that consumed faster than a real client, removing the backpressure needed to produce the failure. Both were withdrawn rather than softened into footnotes.

### The third flattering number needed an external review

The first agent's audit grouped request lengths in blocks, published medians rather than raw rows, and fitted whole-request time against output duration. It wrote the resulting intercept into a document as evidence that the historical per-call floor was wrong by roughly a factor of four. I had no reason to doubt it: the method looked careful, and the conclusion was stated with appropriate hedging.

The second agent overturned it. Request length had changed alongside likely block count, the run order tied length to warm-up drift, and a request-level intercept does not identify a per-block constant at all. Worse, the published medians were insufficient to reproduce the reported intercept exactly — neither agent could recompute it from what the first had published.

Note what did *not* catch this. Not hedging: the original write-up was full of caveats, and none of them were about the thing that was wrong. Not care: the first agent flagged its own small sample. What caught it was a second party re-running the measurement with no stake in the first answer — and, specifically, being told to take its own readings before reading the earlier conclusions. Anchoring is cheap to prevent and expensive to detect afterwards.

The follow-up single-block benchmark fixed the unit-of-analysis error and published all 15 rows. It still did not produce a magic constant: the first ten rows fitted `2.202 + 1.434D`; after five exact repeats in a second order, the fit moved to `2.368 + 1.365D`. Randomization reduces one class of bias. It does not certify a small sample as balanced.

That episode is the strongest evidence in this case study because the process failed before it worked, and because the failure was caught from outside rather than by more diligence from inside. The rule I now hold the work to is concrete: publish the raw rows, not only the summary; preserve the warm-up and inconvenient outliers; separate the party that produces a number from the party that checks it; and build the benchmark so its conclusion can survive the next row.

## What another engineer can take away

When generation is slower than playback and starvation can terminate the response:

1. Measure the client’s actual failure condition before designing the stream.
2. Treat block size as a trade between first-block availability and repeated call overhead.
3. Separate release policy from transport. Earliest and continuous are different objectives.
4. Base underrun protection on media time, not queue emptiness or accidental socket backpressure.
5. Keep body-TTFB, output PCM duration, worker synthesis wall, and device first sound as separate metrics.
6. Regress at the unit named by the claim. Per-call cost needs per-call observations.
7. Preserve every raw row so a third person can recompute the result.

The scheduler did not remove synthesis cost. It placed that cost where the experience could tolerate it, made continuity an explicit policy, and made the remaining uncertainty visible. That is the engineering result.

## Status of the five disputed numbers

| Number | Status |
|---|---|
| **40%** | Withdrawn historical headline. Its denominator had already been invalidated, so it is not a current result. |
| **17.6 s** | Unreviewed historical reading retained only as a measurement failure, never as a baseline. |
| **~10 s** | Retired as a general result. Individual fresh rows land near it, while a separate fixed two-sentence JSONL corpus has body-TTFB median 14.075 s (range 13.268–17.114); sentence count is not a workload definition. |
| **8.2 s** | Unreviewed historical value in the July snapshot’s comments. It was not reproduced and is not used as current behavior. |
| **4.5 s** | Unreviewed historical per-call estimate and simulator setting. The fresh single-block client proxy did not reproduce it as an intercept and cannot establish a replacement worker constant. |
