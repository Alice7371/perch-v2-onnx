# perch-v2-onnx

Minimal, dependency-light inference for **Google Perch v2** ([bioacoustics classifier, ~14,795 species](https://www.kaggle.com/models/google/bird-vocalization-classifier)) from an ONNX export — on Windows, with CUDA, **without installing the CUDA toolkit**.

One script, stdlib + `numpy` + `onnxruntime`. Nothing else.

## Why this exists

Getting Perch v2 running locally hits three walls that cost real time:

1. **Official weights are Kaggle-only** (no official HF repo). The best ONNX export is a community one.
2. **`pip install onnxruntime-gpu` (1.30.x) silently falls back to CPU** on CUDA 12 machines — it wants CUDA 13 DLLs (`cublasLt64_13.dll`). The CUDA 12 line is **1.22.x**.
3. **CUDA/cuDNN DLLs** normally require a system CUDA install — but any `torch` cu12x venv already bundles them, and `os.add_dll_directory()` + `PATH` prepend makes them visible to onnxruntime.

This repo is the distilled working path, verified end-to-end on an RTX 2080.

## Quick start

```bash
uv venv venv --python 3.12
uv pip install --python venv/Scripts/python.exe onnxruntime-gpu==1.22.0 numpy
# CPU only: pip install onnxruntime numpy

# weights (not bundled — see below)
#   perch-v2-fp32.onnx + labels.txt + models.json

python perch_infer.py --cuda-dll-dir "<venv-with-torch>/Lib/site-packages/torch/lib"
# → top-5 + per-window latency
```

Real audio:

```bash
ffmpeg -i recording.wav -ar 32000 -ac 1 clip.wav
python perch_infer.py --audio clip.wav --topk 10 --save-embedding clip_emb.npy
```

## Getting the weights

| Source | What | Notes |
|---|---|---|
| [tphakala/Perch-v2](https://huggingface.co/tphakala/Perch-v2) (HF) | `perch-v2-fp32.onnx`, `labels.txt`, `models.json`, `SHA256SUMS` | community export by the BirdNET-Go author; sha256-verified |
| [google/bird-vocalization-classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier) (Kaggle) | official TF SavedModel (`perch_v2` variation) | use if you want the original TF format |
| [ernensbjorn/perch-v2-int8-tflite](https://huggingface.co/ernensbjorn/perch-v2-int8-tflite), [sammlapp/perch2-tflite](https://huggingface.co/sammlapp/perch2-tflite) | TFLite fp16/int8 | mobile targets |

Verified fingerprint of the export used in testing:

```
sha256  bf0c8467a924cb074663970ca4a0ab1e143602121930209657d0dff5d5cefa1f  perch-v2-fp32.onnx  (409,148,616 bytes)
```

Weights are Google's, Apache-2.0; this repo distributes code only.

## Model spec (from the export's `models.json`)

| | |
|---|---|
| input | `inputs` — `[batch, 160000]` float32, raw waveform @ 32 kHz (5 s window) |
| frontend | PCEN melspectrogram via in-graph DFT (feed audio, not spectrograms) |
| outputs | `embedding` `[b,1536]`, `spatial_embedding` `[b,16,4,1536]`, `spectrogram` `[b,500,128]`, `label` `[b,14795]` |
| labels | `labels.txt` — **line 1 is a header**, then one scientific name per line |

## onnxruntime-gpu ↔ CUDA cheat sheet (the actual trap)

| onnxruntime-gpu | CUDA it wants | on a CUDA 12 box |
|---|---|---|
| 1.30.x (latest, 2026) | CUDA 13 (`cublasLt64_13.dll`), cuDNN 9 | ❌ EP creation fails → **silent CPU fallback** |
| 1.22.x | CUDA 12 (`cublasLt64_12.dll`), cuDNN 9 | ✅ works with torch-bundled DLLs |

Reusing torch's DLLs (no system CUDA install):

```python
os.add_dll_directory(torch_lib_dir)                    # e.g. .../site-packages/torch/lib
os.environ["PATH"] = torch_lib_dir + os.pathsep + os.environ["PATH"]
```

Or just pass it to the script: `--cuda-dll-dir .../torch/lib`.

## Measured performance (RTX 2080 8 GB, fp32, 5 s window)

| provider | per window | realtime factor |
|---|---|---|
| CUDA (1.22.0) | **41 ms** | 121× |
| CPU | 81 ms | 62× |

GPU and CPU produce identical top-5 on a 440 Hz test tone (Telephone / Alarm / Glass … — a pure tone is, sensibly, a telephone).
