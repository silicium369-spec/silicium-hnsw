# Silicium Vector Search Evaluation Suite (x86_64 AVX-512 & AVX2)

High-throughput HNSW vector search engine written in Rust with runtime hardware dispatch (`AVX-512 VNNI/FMA` and `AVX2 FMA`), cache-aligned contiguous graph routing, and quantized `INT8 SQ8` graph routing with exact `FP32` top-pool reranking (`4.0x` primary vector memory compression).

## Quick Start

```bash
# 1. Verify SHA-256 digests, C-ABI bindings, and 1-vCPU head-to-head confrontation
./reproduce_all.sh

# 2. Download and evaluate full TexMex SIFT-1M (1,000,000 vectors, 128D, 10,000 queries)
./reproduce_all.sh --sift1m

# 3. Evaluate High-Dimensional embeddings (Cohere-768D & OpenAI-1536D)
./reproduce_all.sh --high-dim
```

To compare directly against third-party engines (`usearch`, `hnswlib`, `faiss-cpu`) on the same CPU core (`taskset -c 0`):

```bash
pip install numpy pyyaml usearch hnswlib faiss-cpu
python3 confrontation_arena.py --num-vectors 100000 --num-queries 1000 --ef 16,32,64
```

### Isolated Evaluation (Hermetic Docker Sandbox)

To evaluate the pre-compiled C-ABI library inside an ephemeral, read-only Docker container without installing dependencies or running binaries directly on your host OS:

```bash
docker run --rm -it -v "$PWD":/workspace:ro -w /tmp/bench python:3.11-slim bash -c \
  "cp -r /workspace/* . && apt-get update -qq && apt-get install -y -qq curl file util-linux && \
   pip install -q numpy pyyaml usearch hnswlib faiss-cpu && ./reproduce_all.sh"
```

## 1. SIFT-128D Benchmark Results (AWS `c7a.2xlarge` — AMD EPYC 9R14, Zen 4 AVX-512)

### Multi-Engine 1-vCPU Confrontation (SIFT-128D, `k=10`, `M=16`, `efConstruction=100`, `taskset -c 0`)

- **SIFT-100K 1-vCPU Index Build Time**: **5.24 s** (`Silicium`) vs `5.93 s` (`usearch`), `7.71 s` (`hnswlib`), `9.32 s` (`FAISS`)

| Dataset & `efSearch` | Silicium (`QPS` / `Recall@10`) | `usearch` (`QPS` / `Recall@10`) | `hnswlib` (`QPS` / `Recall@10`) | `FAISS` (`QPS` / `Recall@10`) | Speedup vs `usearch` / `hnswlib` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SIFT-10K** (`ef=12`) | **190,281 QPS** (`0.9931`) | 146,653 QPS (`0.9701`) | 134,801 QPS (`0.9705`) | 135,261 QPS (`0.9689`) | **1.30x** / **1.41x** |
| **SIFT-10K** (`ef=16`) | **155,991 QPS** (`0.9979`) | 122,131 QPS (`0.9867`) | 114,179 QPS (`0.9870`) | 116,216 QPS (`0.9875`) | **1.28x** / **1.37x** |
| **SIFT-10K** (`ef=24`) | **118,027 QPS** (`1.0000`) | 94,813 QPS (`0.9972`) | 89,588 QPS (`0.9971`) | 89,930 QPS (`0.9978`) | **1.24x** / **1.32x** |
| **SIFT-100K** (`ef=16`) | **100,129 QPS** (`0.9223`) | 49,040 QPS (`0.9002`) | 35,582 QPS (`0.9069`) | 33,588 QPS (`0.8885`) | **2.04x** / **2.81x** |
| **SIFT-100K** (`ef=32`) | **55,835 QPS** (`0.9736`) | 29,534 QPS (`0.9696`) | 21,799 QPS (`0.9720`) | 19,595 QPS (`0.9640`) | **1.89x** / **2.56x** |
| **SIFT-100K** (`ef=64`) | **32,187 QPS** (`0.9909`) | 17,466 QPS (`0.9937`) | 13,356 QPS (`0.9917`) | 11,508 QPS (`0.9868`) | **1.84x** / **2.41x** |
| **SIFT-100K** (`ef=128`) | **19,027 QPS** (`0.9963`) | 10,644 QPS (`0.9967`) | 8,385 QPS (`0.9975`) | 6,968 QPS (`0.9985`) | **1.79x** / **2.27x** |

### Official SIFT-1M Full-Scale Evaluation (1,000,000 vectors, 128D, 10,000 queries, `k=10`)

- **Default `balanced` Profile (`M=24, efC=160`)**: `27.79 s` 8-vCPU index build (`35,985 vectors/s`), **85.69% Recall@10** at `ef=16` (`294,476 QPS`) up to **99.81% Recall@10** at `ef=256`
- **High-Throughput `fast_stream` Profile (`M=16, efC=100`)**: `14.49 s` 8-vCPU index build (`68,994 vectors/s`), **369,836 QPS** peak throughput

| `ef` | `Recall@10` (`balanced` `M=24`) | 1-Core / 8-Core `QPS` (`balanced`) | 1-Core `P50` / `P99` (`balanced`) | `Recall@10` (`fast_stream` `M=16`) | 1-Core / 8-Core `QPS` (`fast_stream`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `16` | **0.8569** (`+9.51%`) | **39,540 QPS** / **294,476 QPS** | `24.91µs` / `40.45µs` | `0.7618` | **49,261 QPS** / **369,836 QPS** |
| `24` | **0.9078** (`+7.75%`) | **29,404 QPS** / **219,191 QPS** | `33.89µs` / `50.81µs` | `0.8303` | **36,902 QPS** / **278,597 QPS** |
| `32` | **0.9362** (`+6.44%`) | **23,549 QPS** / **175,582 QPS** | `42.70µs` / `60.07µs` | `0.8718` | **29,787 QPS** / **224,878 QPS** |
| `48` | **0.9649** (`+4.71%`) | **16,873 QPS** / **125,184 QPS** | `60.00µs` / `79.60µs` | `0.9178` | **21,640 QPS** / **162,608 QPS** |
| `64` | **0.9777** (`+3.52%`) | **13,260 QPS** / **99,105 QPS** | `77.21µs` / `99.48µs` | `0.9425` | **17,101 QPS** / **128,433 QPS** |
| `128` | **0.9936** (`+1.49%`) | **7,316 QPS** / **54,599 QPS** | `140.86µs` / `174.26µs` | `0.9787` | **9,374 QPS** / **71,169 QPS** |
| `256` | **0.9981** (`+0.52%`) | **3,925 QPS** / **29,393 QPS** | `262.42µs` / `322.18µs` | `0.9929` | **5,048 QPS** / **38,223 QPS** |

## 2. High-Dimensional Embeddings (`OpenAI-1536D` & `Cohere-768D`, 25,000 vectors, 1,000 queries)

Quantized `INT8 SQ8` graph routing with exact `FP32` reranking (`balanced` profile, `M=24, efConstruction=160`) reduces primary vector memory by **4.0x** (`36.62 MB` vs `146.48 MB` at `1536D`; `18.31 MB` vs `73.24 MB` at `768D`) while accelerating both index construction and query throughput on `AMD EPYC 9R14` (AWS `c7a.2xlarge`, 8 vCPUs, Zen 4 AVX-512):

| Workload & Metric (`M=24, efC=160`) | `ef` | `Recall@10` | 1-Core `QPS` (`P50` / `P99`) | 8-Core `QPS` | Build (`SQ8` vs `FP32`) | Query Speedup (8c) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenAI-1536D** (`Cosine`) | `16` | **0.9761** | **19,935 QPS** (`46.8µs` / `85.3µs`) | **145,480 QPS** | `0.999s` vs `3.811s` (**3.81x**) | **3.50x** |
| **OpenAI-1536D** (`Cosine`) | `32` | **0.9930** | **13,626 QPS** (`70.8µs` / `119.4µs`) | **101,774 QPS** | `0.999s` vs `3.811s` (**3.81x**) | **3.32x** |
| **OpenAI-1536D** (`Cosine`) | `64` | **1.0000** | **10,322 QPS** (`93.1µs` / `149.6µs`) | **77,854 QPS** | `0.999s` vs `3.811s` (**3.81x**) | **3.35x** |
| **Cohere-768D** (`Dot`) | `16` | **0.9746** | **42,351 QPS** (`22.7µs` / `41.6µs`) | **312,248 QPS** | `0.585s` vs `1.681s` (**2.87x**) | **3.20x** |
| **Cohere-768D** (`Dot`) | `32` | **0.9930** | **26,544 QPS** (`35.8µs` / `63.1µs`) | **194,587 QPS** | `0.585s` vs `1.681s` (**2.87x**) | **2.41x** |
| **Cohere-768D** (`Dot`) | `64` | **0.9980** | **18,729 QPS** (`51.2µs` / `84.7µs`) | **141,861 QPS** | `0.585s` vs `1.681s` (**2.87x**) | **2.49x** |

## Repository Structure & Legal

- `Dockerfile`, `config.yml`, `module.py`: Standard `ann-benchmarks` container and C-ABI adapter.
- `confrontation_arena.py`, `reproduce_all.sh`, `scripts/download_release.sh`: Standalone verification and multi-engine benchmark harness.
- `results/`: Certified AWS `c7a.2xlarge` (`AMD EPYC 9R14`) telemetry JSON artifacts (`confrontation_aws_c7a_results.json`, `sift1m_aws_c7a_results.json`, `high_dim_angular_dot_product_aws_c7a_results.json`).
- `LICENSE`, `PATENTS.md`, `NOTICE`, `SECURITY.md`, `checksums.sha256`: Evaluation license terms (SCSL-1.0 / PolyForm Noncommercial 1.0.0, Patent Pending) and SHA-256 integrity manifest.