# Host experiment summary

Measured on 11 October 2026 using the GitHub Actions host validation workflow (Ubuntu 24.04, GCC 13.3.0). The Debug, Release and AddressSanitizer/UndefinedBehaviorSanitizer jobs passed.

## Test coverage

- 3 CTest suites passed: scheduler (14 named cases), KV allocator (13 named cases), and stress tests (2 named cases).
- The same host benchmark and experiment checks were run in Debug, Release, and ASan/UBSan builds.
- Shell syntax, Python syntax, repository links, and SVG readability checks passed.

## Experiment size

- 36 scheduler settings: 3 batch sizes (4, 8, 16) × 3 token budgets (32, 64, 128) × 4 prefill chunks (8, 16, 32, 64).
- 108 scheduler records across decode-heavy, mixed-long-prompt, and prompt-burst workloads.
- 16 KV-cache records across short, mixed, ragged and long sequence patterns.

## Lowest p95 decode-gap setting for each request mix

| Workload | Batch | Token budget | Prefill chunk | p95 decode gap (scheduler iterations) | Total scheduler iterations |
|---|---:|---:|---:|---:|---:|
| decode_heavy | 16 | 32 | 8 | 2.000 | 160 |
| mixed_long_prompt | 16 | 128 | 64 | 2.000 | 168 |
| prompt_burst | 4 | 128 | 64 | 1.000 | 131 |

These settings are selected first by the lowest p95 gap, then by fewer total scheduler iterations, then by the smaller chunk size. These iteration counts are simulated scheduling steps; they are not wall-clock latency.

## KV-cache allocation results

| Workload | Page size | Logical tokens | Reserved tokens | Reservation reduction | Internal fragmentation |
|---|---:|---:|---:|---:|---:|
| long | 8 | 20736 | 20736 | 36.72% | 0.00% |
| long | 16 | 20736 | 20736 | 36.72% | 0.00% |
| long | 32 | 20736 | 20736 | 36.72% | 0.00% |
| long | 64 | 20736 | 20736 | 36.72% | 0.00% |
| mixed | 8 | 3392 | 3392 | 89.65% | 0.00% |
| mixed | 16 | 3392 | 3392 | 89.65% | 0.00% |
| mixed | 32 | 3392 | 3392 | 89.65% | 0.00% |
| mixed | 64 | 3392 | 3392 | 89.65% | 0.00% |
| ragged | 8 | 3387 | 3416 | 89.58% | 0.85% |
| ragged | 16 | 3387 | 3440 | 89.50% | 1.54% |
| ragged | 32 | 3387 | 3488 | 89.36% | 2.90% |
| ragged | 64 | 3387 | 3584 | 89.06% | 5.50% |
| short | 8 | 552 | 552 | 98.32% | 0.00% |
| short | 16 | 552 | 560 | 98.29% | 1.43% |
| short | 32 | 552 | 608 | 98.14% | 9.21% |
| short | 64 | 552 | 704 | 97.85% | 21.59% |

## Findings

- Prompt-burst cases had a p95 decode gap of one scheduler iteration across the tested settings. This is a scheduler simulation result, not real inter-token latency.
- Decode-heavy cases had a best p95 gap of two iterations because the number of active requests exceeded the configured batch limit.
- Mixed long prompts reached a best p95 gap of two iterations. Larger batch limits reduced the number of scheduler iterations in this simulation, but the counts are not wall-clock time.
- For short sequences, reserved tokens increased from 552 with page size 8 to 704 with page size 64. Internal fragmentation rose from 0% to 21.59%.
- For ragged sequence lengths, internal fragmentation rose from 0.85% at page size 8 to 5.50% at page size 64.
- The mixed and long workloads were aligned to all four page sizes, so those cases had no page-rounding waste in this experiment.

## Scope

These are deterministic host scheduling simulations and KV allocation calculations. They do not measure GPU kernel time, end-to-end inter-token latency, or total process VRAM. CUDA attention and TensorRT-LLM serving benchmarks still need a compatible NVIDIA system.
