"""Minimal runnable inference for Google Perch v2 ONNX on Windows/Linux.

Model: Perch v2 — bioacoustics classifier, 14,795 species (birds, insects,
amphibians, mammals). Input is raw float32 waveform [batch, 160000] @ 32 kHz
(5 s), the in-graph DFT frontend means you feed audio, not spectrograms.

Weights are NOT bundled here — see README for download sources.

Usage:
    python perch_infer.py                          # synthesized 440 Hz test tone
    python perch_infer.py --audio clip.wav --topk 10
    python perch_infer.py --audio clip.wav --save-embedding emb.npy
    python perch_infer.py --cpu                    # force CPUExecutionProvider
    python perch_infer.py --cuda-dll-dir "C:/path/to/torch/lib"
"""
from __future__ import annotations

import argparse
import os
import time
import wave

import numpy as np

SAMPLE_RATE = 32000
WINDOW = 160000  # 5 s @ 32 kHz


def load_audio(path: str) -> np.ndarray:
    """Read a 16-bit PCM mono 32 kHz WAV as float32 [-1, 1)."""
    with wave.open(path, "rb") as w:
        if w.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{path}: sample rate is {w.getframerate()} Hz, need 32000 Hz "
                "(resample first, e.g. ffmpeg -i in.wav -ar 32000 -ac 1 out.wav)"
            )
        if w.getnchannels() != 1:
            raise SystemExit(
                f"{path}: {w.getnchannels()} channels, need mono "
                "(-ac 1 in ffmpeg)"
            )
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16-bit PCM WAV")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def synth_test_tone() -> np.ndarray:
    t = np.arange(WINDOW, dtype=np.float32) / SAMPLE_RATE
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    noise = 0.05 * np.random.default_rng(0).standard_normal(WINDOW).astype(np.float32)
    return tone + noise


def to_windows(audio: np.ndarray) -> np.ndarray:
    """Split into non-overlapping 5 s windows, zero-padding the tail."""
    if len(audio) < WINDOW:
        padded = np.zeros(WINDOW, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded[None, :]
    n = len(audio) // WINDOW
    out = audio[: n * WINDOW].reshape(n, WINDOW)
    if len(audio) % WINDOW:
        tail = np.zeros(WINDOW, dtype=np.float32)
        tail[: len(audio) - n * WINDOW] = audio[n * WINDOW :]
        out = np.concatenate([out, tail[None, :]], axis=0)
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="perch-v2-fp32.onnx")
    ap.add_argument("--labels", default="labels.txt")
    ap.add_argument("--audio", default=None, help="16-bit PCM mono 32 kHz WAV")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--save-embedding", default=None, help="save 1536-d embedding(s) to .npy")
    ap.add_argument("--cpu", action="store_true", help="force CPUExecutionProvider")
    ap.add_argument(
        "--cuda-dll-dir",
        default=None,
        help="directory holding CUDA 12 + cuDNN 9 DLLs (e.g. a torch/lib "
        "folder) — lets onnxruntime-gpu run without a system CUDA install",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit(f"model not found: {args.model} — see README for download sources")
    if not os.path.isfile(args.labels):
        raise SystemExit(f"labels not found: {args.labels}")

    # CUDA EP resolves its DLLs through PATH; add_dll_directory alone is not enough.
    if args.cuda_dll_dir and not args.cpu:
        os.add_dll_directory(args.cuda_dll_dir)
        os.environ["PATH"] = args.cuda_dll_dir + os.pathsep + os.environ.get("PATH", "")

    import onnxruntime as ort

    providers = ["CPUExecutionProvider"] if args.cpu else [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(args.model, providers=providers)
    active = sess.get_providers()[0]
    print(f"onnxruntime {ort.__version__} | active provider: {active}")

    iname = sess.get_inputs()[0].name
    onames = [o.name for o in sess.get_outputs()]

    # labels.txt line 1 is a header ("inat2024_fsd50k"), then one species per line
    lines = open(args.labels, encoding="utf-8").read().splitlines()
    class_names = lines[1:]

    audio = load_audio(args.audio) if args.audio else synth_test_tone()
    x = to_windows(audio)
    print(f"input: {x.shape[0]} window(s) x 5 s")

    outs = sess.run(onames, {iname: x})
    by_name = dict(zip(onames, outs))
    logits = by_name["label"] if "label" in by_name else outs[0]

    for w in range(x.shape[0]):
        p = softmax(logits[w])
        top = np.argsort(p)[::-1][: args.topk]
        src = args.audio or "synth 440 Hz tone"
        print(f"\nwindow {w} ({src}) top-{args.topk}:")
        for i in top:
            print(f"  {p[i] * 100:6.2f}%  {class_names[i]}")

    if args.save_embedding:
        np.save(args.save_embedding, by_name.get("embedding", outs[0]).astype(np.float32))
        print(f"\nembedding saved: {args.save_embedding}")

    # latency (single-window timing)
    one = x[:1]
    sess.run(onames, {iname: one})
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        sess.run(onames, {iname: one})
        times.append(time.perf_counter() - t0)
    mean = float(np.mean(times))
    print(f"\nper-window latency: mean {mean * 1000:.0f} ms  ({5.0 / mean:.0f}x realtime)")


if __name__ == "__main__":
    main()
