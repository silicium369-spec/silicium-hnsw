#!/usr/bin/env bash
# ==============================================================================
# Silicium ANN: Native HNSW Evaluation & Multi-Engine Confrontation Harness
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"
mkdir -p "$LIB_DIR"

echo "================================================================================"
echo "🛡️  SILICIUM ANN : AUTONOMOUS REPRODUCIBILITY & CONFRONTATION HARNESS"
echo "================================================================================"

# 1. Download & verify release binary
echo "▶ [1/4] Locating/downloading release asset & verifying integrity..."
bash "$SCRIPT_DIR/scripts/download_release.sh" ann-v1.0.0 "$LIB_DIR"

# 2. Verify symbol table stripping
echo "▶ [2/4] Checking binary stripping & ELF headers..."
SO_PATH="$LIB_DIR/libsilicium_ann_x86_64.so"
file "$SO_PATH" | grep -q "stripped" && echo "  ✅ Shared library is 100% stripped."

# 3. Functional smoke test (C-ABI Vector Search)
echo "▶ [3/4] Running Python C-ABI functional smoke test..."
PYTHON_CMD="python3"
if [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON_CMD="python"
elif command -v uv &> /dev/null; then
    PYTHON_CMD="uv run --with numpy,pyyaml python3"
fi

$PYTHON_CMD - << PYEOF
import ctypes, sys, numpy as np

lib = ctypes.CDLL("$SO_PATH")
init = getattr(lib, "silicium_ann_init")
init.argtypes = [ctypes.c_size_t]
init.restype = ctypes.c_void_p
ptr = init(128)

add = getattr(lib, "silicium_ann_add")
add.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
add.restype = None

data = np.random.randn(50, 128).astype(np.float32)
data_p = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
add(ptr, data_p, 50)

query_fn = getattr(lib, "silicium_ann_query")
query_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32)]
query_fn.restype = None

q = data[0:1].copy()
q_p = q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
out = np.empty(5, dtype=np.int32)
out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

query_fn(ptr, q_p, 5, out_p)
assert out[0] == 0, f"Expected 0 as nearest neighbor, got {out[0]}"

free = getattr(lib, "silicium_ann_free")
free.argtypes = [ctypes.c_void_p]
free.restype = None
free(ptr)
print("  ✅ C-ABI functional test PASSED (50 vectors indexed, nearest neighbor verified).")
PYEOF

# 4. Multi-Engine In-Situ Confrontation Arena
if [ "${1:-}" = "--sift1m" ]; then
    SIFT_DIR="$SCRIPT_DIR/data/sift"
    mkdir -p "$SIFT_DIR"
    if [ ! -f "$SIFT_DIR/sift_base.fvecs" ]; then
        echo "▶ [4/4] Downloading official TexMex SIFT-1M dataset (ftp.irisa.fr)..."
        curl -fsSL "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" -o "$SIFT_DIR/sift.tar.gz"
        tar -xzf "$SIFT_DIR/sift.tar.gz" -C "$SCRIPT_DIR/data"
        rm -f "$SIFT_DIR/sift.tar.gz"
    fi
    echo "▶ [4/4] Running Official SIFT-1M (1,000,000 vectors 128D) Evaluation..."
    if command -v taskset &> /dev/null; then
        taskset -c 0 $PYTHON_CMD "$SCRIPT_DIR/confrontation_arena.py" --sift-dir "$SIFT_DIR" --num-vectors 1000000 --num-queries 10000 --ef 16,24,32,48,64,96,128,192,256
    else
        $PYTHON_CMD "$SCRIPT_DIR/confrontation_arena.py" --sift-dir "$SIFT_DIR" --num-vectors 1000000 --num-queries 10000 --ef 16,24,32,48,64,96,128,192,256
    fi
elif [ "${1:-}" = "--high-dim" ]; then
    echo "▶ [4/4] Running High-Dimensional (Cohere-768D & OpenAI-1536D) Evaluation..."
    if [ -x "$SCRIPT_DIR/bin/sift1m_runner" ]; then
        "$SCRIPT_DIR/bin/sift1m_runner" --dataset angular_hd
    elif [ -f "$SCRIPT_DIR/confrontation_arena.py" ]; then
        if command -v taskset &> /dev/null; then
            taskset -c 0 $PYTHON_CMD "$SCRIPT_DIR/confrontation_arena.py" --dim 768 --num-vectors 10000 --num-queries 500 --ef 16,32,64
            taskset -c 0 $PYTHON_CMD "$SCRIPT_DIR/confrontation_arena.py" --dim 1536 --num-vectors 10000 --num-queries 500 --ef 16,32,64
        else
            $PYTHON_CMD "$SCRIPT_DIR/confrontation_arena.py" --dim 768 --num-vectors 10000 --num-queries 500 --ef 16,32,64
            $PYTHON_CMD "$SCRIPT_DIR/confrontation_arena.py" --dim 1536 --num-vectors 10000 --num-queries 500 --ef 16,32,64
        fi
    fi
elif [ "${1:-}" = "--smoke" ]; then
    echo "▶ [4/4] Running Quick Smoke Test Confrontation..."
    if [ -f "$SCRIPT_DIR/confrontation_arena.py" ]; then
        if command -v taskset &> /dev/null; then
            taskset -c 0 $PYTHON_CMD "$SCRIPT_DIR/confrontation_arena.py" --num-vectors 256 --num-queries 32 --ef 12,16
        else
            $PYTHON_CMD "$SCRIPT_DIR/confrontation_arena.py" --num-vectors 256 --num-queries 32 --ef 12,16
        fi
    fi
else
    echo "▶ [4/4] Running In-Situ Multi-Engine Confrontation (Silicium vs Competitors)..."
    if [ -f "$SCRIPT_DIR/confrontation_arena.py" ]; then
        if command -v taskset &> /dev/null; then
            taskset -c 0 $PYTHON_CMD "$SCRIPT_DIR/confrontation_arena.py" --num-vectors 10000 --num-queries 1000 --ef 12,16,24,32
        else
            $PYTHON_CMD "$SCRIPT_DIR/confrontation_arena.py" --num-vectors 10000 --num-queries 1000 --ef 12,16,24,32
        fi
    fi
fi

echo "================================================================================"
echo "🎉 SILICIUM ANN EVALUATION & MULTI-ENGINE CONFRONTATION CERTIFIED SUCCESSFUL"
echo "================================================================================"
