# Content observer performance — 2026-09-12

The main additional cost is the viewport rescan triggered by a relevant mutation. The observer callback itself is small in these cases; offscreen Chinese text updates can nevertheless cause substantial main-thread work on larger documents.

## Measured results

Actual refresh scans after six offscreen Chinese-text edits spaced 300 ms apart. Each row aggregates three repetitions; scan median/p95 use 18 samples, and total work is the mean per six-edit run.

| Offscreen Chinese text nodes | Median scan | p95 scan | Scans per run | Total measured work per run | Additional process calls |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 6.9 ms | 19.1 ms | 6 | 42.7 ms | 0 |
| 1,000 | 21.1 ms | 22.9 ms | 6 | 126.7 ms | 0 |
| 5,000 | 102.9 ms | 113.7 ms | 6 | 605.9 ms | 0 |

- Observer OFF: zero automatic scans and zero additional process calls for these fixed-layout offscreen updates.
- Idle: zero observer callbacks and zero refresh scans during each approximately 850 ms observation window.
- Forty numeric counter updates at a requested 25 ms interval: zero refresh scans; approximately 1.0–1.2 ms of callback work in total.
- Forty Chinese-text updates at a requested 25 ms interval: forty refresh requests coalesced into one final scan. Observer callbacks totaled approximately 1.4–1.9 ms per run, excluding the final scan.
- Six Chinese-text updates at 300 ms intervals: six separate scans; the 200 ms debounce does not merge these.
- All measured updates were offscreen. No measured scenario invoked the processing dependency after the initial visible text was handled.

The 5,000-node case spends roughly 100 ms in a synchronous scan. That is large enough to disrupt smooth interaction. The optimization target is preventing unnecessary full reads after offscreen-only changes.

## Method and limits

- Code: `6be0ca4`, with runtime files unchanged. Browser benchmark in [content-performance.html](content-performance.html); [raw measurements](content-performance-results.json).
- Apple M5 Pro, macOS 26.6.2; Chrome reports version 152.0.0.0.
- Measurements ran from 00:37:57 to 00:40:06 Asia/Shanghai on 2026-09-12. The test tab stayed visible.
- The same fixture was tested with the new content observer enabled and disabled; the body-size/viewport watcher remained enabled in both. No viewport events occurred during measurements.
- Three repetitions at each size; ON/OFF order alternated. Each fixture had eight already-processed visible paragraphs, two nested open shadow roots, and 300, 1,000, or 5,000 unprocessed offscreen Chinese text nodes (one paragraph/span pair per node).
- The real reader, DOM traversal, visibility checks, processed-text tracking, marker writing, observer, and debounce ran in Chrome. Inference was stubbed deliberately to isolate DOM overhead.
- The fixture runs in a same-origin child frame, preventing the installed extension, which only injects into frame 0, from processing the same content a second time.
- Timings use performance.now() around observer callbacks, reads, and writes. Total measured work is their sum; it is not whole-browser CPU usage. Native observer bookkeeping, unrelated browser work, rendering, initial setup, and cleanup are excluded. No memory profiling was performed.
- A separate warmed-read microbenchmark is preserved in the raw data: three warm-up reads followed by fifteen measured reads per run. The table above uses actual refresh scans after mutations, not that microbenchmark.
- This is a controlled workload, not a live Bilibili CPU profile. The loaded target page had 2,548 elements, 1,765 text nodes, 292 Han-containing text nodes, and two open shadow roots before scrolling. Its structure and real update frequency differ from the fixture; the 300-node result is context, not a Bilibili latency estimate.

## Reproduce

Run `python3 -m http.server 8765 --bind 127.0.0.1` from the repository, open `http://127.0.0.1:8765/test/content-performance.html`, click **Run benchmark**, and keep the tab visible. Completion displays the full JSON, including validity checks. Expect approximately two minutes.
