from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

KERNEL_CATS = {"kernel", "Kernel"}
COPY_CATS = {"gpu_memcpy", "gpu_memset", "Memcpy", "Memset", "MemcpyH2D", "MemcpyD2H"}

KIND_KERNEL, KIND_H2D, KIND_D2H, KIND_D2D, KIND_MEMSET = "kernel", "h2d", "d2h", "d2d", "memset"
TRANSFER_KINDS = (KIND_H2D, KIND_D2H)
DEVICE_WORK_KINDS = (KIND_KERNEL, KIND_D2D, KIND_MEMSET)


def classify_event(name: str) -> str:
    n = name.lower().replace(" ", "")
    if "memset" in n:
        return KIND_MEMSET
    if "memcpy" in n:
        if "htod" in n or "h2d" in n:
            return KIND_H2D
        if "dtoh" in n or "d2h" in n:
            return KIND_D2H
        if "dtod" in n or "d2d" in n:
            return KIND_D2D
        return KIND_D2D
    return KIND_KERNEL


def union_duration(intervals) -> float:
    iv = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not iv:
        return 0.0
    total = 0.0
    cur_start, cur_end = iv[0]
    for s, e in iv[1:]:
        if s > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = s, e
        elif e > cur_end:
            cur_end = e
    total += cur_end - cur_start
    return total


def union_gaps(intervals, t0=None, t1=None):
    iv = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not iv:
        return [(t0, t1)] if (t0 is not None and t1 is not None and t1 > t0) else []
    merged = [list(iv[0])]
    for s, e in iv[1:]:
        if s > merged[-1][1]:
            merged.append([s, e])
        elif e > merged[-1][1]:
            merged[-1][1] = e
    gaps = []
    if t0 is not None and merged[0][0] > t0:
        gaps.append((t0, merged[0][0]))
    for i in range(len(merged) - 1):
        gaps.append((merged[i][1], merged[i + 1][0]))
    if t1 is not None and merged[-1][1] < t1:
        gaps.append((merged[-1][1], t1))
    return gaps


def summarize(kernels, copies, wall_time_s):
    by_kind = {k: [] for k in (KIND_KERNEL, KIND_H2D, KIND_D2H, KIND_D2D, KIND_MEMSET)}
    for s, e, name, _st in kernels:
        by_kind[KIND_KERNEL].append((s, e))
    for s, e, name, _st in copies:
        by_kind[classify_event(name)].append((s, e))

    k_iv = by_kind[KIND_KERNEL]
    device_iv = [iv for k in DEVICE_WORK_KINDS for iv in by_kind[k]]
    transfer_iv = [iv for k in TRANSFER_KINDS for iv in by_kind[k]]

    busy_us = union_duration(k_iv)
    device_us = union_duration(device_iv)
    transfer_us = union_duration(transfer_iv)
    any_us = union_duration(device_iv + transfer_iv)
    busy_s = busy_us / 1e6

    out = {
        "wall_time_s": wall_time_s,
        "busy_time_s": busy_s,
        "device_work_time_s": device_us / 1e6,
        "transfer_time_s": transfer_us / 1e6,
        "gpu_active_time_s": any_us / 1e6,
        "cpu_wait_time_s": max(0.0, wall_time_s - any_us / 1e6),
        "kernel_sum_time_s": sum(e - s for s, e in k_iv) / 1e6,
        "overlap_factor": (sum(e - s for s, e in k_iv) / busy_us) if busy_us > 0 else float("nan"),
        "n_kernels": len(kernels),
        "n_copies": len(copies),
        "gpu_busy_pct": (100.0 * busy_s / wall_time_s) if wall_time_s > 0 else float("nan"),
    }
    for k in (KIND_H2D, KIND_D2H, KIND_D2D, KIND_MEMSET):
        out[f"n_{k}"] = len(by_kind[k])
        out[f"{k}_time_s"] = union_duration(by_kind[k]) / 1e6
    return out


class _TorchBackend:
    name = "torch"

    @staticmethod
    def available() -> bool:
        try:
            import torch
            return torch.cuda.is_available() and hasattr(torch, "profiler")
        except Exception:
            return False

    def __init__(self):
        import torch
        self._torch = torch
        self._prof = None

    def start(self):
        from torch.profiler import ProfilerActivity, profile
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        )
        self._prof.__enter__()

    def stop(self, tmp_path: Path):
        self._torch.cuda.synchronize()
        self._prof.__exit__(None, None, None)
        self._prof.export_chrome_trace(str(tmp_path))
        self._prof = None
        return _parse_chrome_trace(tmp_path)


def _parse_chrome_trace(path: Path):
    with open(path) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    kernels, copies = [], []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        dur = ev.get("dur")
        ts = ev.get("ts")
        if dur is None or ts is None:
            continue
        stream = (ev.get("args") or {}).get("stream", -1)
        rec = (float(ts), float(ts) + float(dur), ev.get("name", ""), stream)
        if cat in KERNEL_CATS:
            kernels.append(rec)
        elif cat in COPY_CATS:
            copies.append(rec)
    return kernels, copies


class _TFBackend:
    name = "tensorflow"

    @staticmethod
    def available() -> bool:
        try:
            import tensorflow as tf
            from tensorflow.core.profiler.protobuf import xplane_pb2
            return True
        except Exception:
            return False

    def __init__(self):
        import tensorflow as tf
        self._tf = tf
        self._logdir = None

    def start(self):
        import tempfile
        self._logdir = tempfile.mkdtemp(prefix="hcpa_tfprof_")
        self._tf.profiler.experimental.start(self._logdir)

    def stop(self, tmp_path: Path):
        self._tf.profiler.experimental.stop()
        try:
            return _parse_xplane(self._logdir)
        finally:
            self._logdir = None


def _parse_xplane(logdir: str):
    from tensorflow.core.profiler.protobuf import xplane_pb2

    pbs = sorted(Path(logdir).rglob("*.xplane.pb"))
    kernels, copies = [], []
    for pb in pbs:
        space = xplane_pb2.XSpace()
        space.ParseFromString(pb.read_bytes())
        for plane in space.planes:
            if "GPU" not in plane.name or "device" not in plane.name.lower():
                continue
            names = {m.id: m.name for m in plane.event_metadata.values()}
            for line in plane.lines:
                lname = (line.name or "").lower()
                if not lname.startswith("stream"):
                    continue
                for ev in line.events:
                    ename = names.get(ev.metadata_id, "")
                    el = ename.lower()
                    is_copy = "memcpy" in el or "memset" in el
                    start_us = (line.timestamp_ns + ev.offset_ps / 1e3) / 1e3
                    dur_us = ev.duration_ps / 1e6
                    if dur_us <= 0:
                        continue
                    rec = (start_us, start_us + dur_us, ename, line.id)
                    (copies if is_copy else kernels).append(rec)
    return kernels, copies


def _parse_epoch_list(raw):
    out = set()
    for tok in (raw or "").replace(";", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.add(int(tok))
    return out


class KernelProfiler:

    def __init__(self, approach: str, out_dir=None, epochs=None, backend=None):
        self.approach = approach
        self.epochs = epochs if epochs is not None else _parse_epoch_list(
            os.environ.get("HCPA_PROFILE_EPOCHS", ""))
        self.out_dir = Path(out_dir or os.environ.get("HCPA_PROFILE_DIR", "./kprof"))
        self.last_summary = None
        self._backend = None
        self._active = None
        if not self.epochs:
            return
        for cls in (backend,) if backend else (_TorchBackend, _TFBackend):
            try:
                if cls.available():
                    self._backend = cls()
                    break
            except Exception:
                continue
        if self._backend is None:
            print("[kprof] nenhum backend de profiler disponível; desativado")
            self.epochs = set()

    @property
    def enabled(self) -> bool:
        return bool(self.epochs) and self._backend is not None

    def should_profile(self, epoch_idx: int) -> bool:
        return self.enabled and epoch_idx in self.epochs

    def begin(self, epoch_idx: int) -> bool:
        if not self.should_profile(epoch_idx) or self._active is not None:
            return False
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._backend.start()
        except Exception as exc:
            print(f"[kprof] falha ao iniciar (epoch {epoch_idx}): {exc}")
            return False
        self._active = (epoch_idx, time.perf_counter())
        return True

    def end(self):
        if self._active is None:
            return None
        epoch_idx, t0 = self._active
        self._active = None
        wall = time.perf_counter() - t0
        tmp = self.out_dir / f"_trace_{epoch_idx}.json"
        try:
            kernels, copies = self._backend.stop(tmp)
            summary = summarize(kernels, copies, wall)
            summary.update(epoch=epoch_idx, approach=self.approach,
                           backend=self._backend.name)
            self.last_summary = summary
            payload = {
                "summary": summary,
                "kernels": [[s, e, st] for s, e, _n, st in kernels],
                "copies": [[s, e, st, classify_event(n)] for s, e, n, st in copies],
            }
            out = self.out_dir / f"epoch_{epoch_idx}.json"
            out.write_text(json.dumps(payload))
            print(f"[kprof] epoch {epoch_idx}: busy={summary['busy_time_s']:.2f}s "
                  f"cpu_wait={summary['cpu_wait_time_s']:.2f}s "
                  f"kernels={summary['n_kernels']} -> {out}")
            return summary
        except Exception as exc:
            print(f"[kprof] falha ao coletar (epoch {epoch_idx}): {exc}")
            return None
        finally:
            tmp.unlink(missing_ok=True)

    @contextmanager
    def epoch(self, epoch_idx: int):
        started = self.begin(epoch_idx)
        try:
            yield self if started else None
        finally:
            if started:
                self.end()

    def keras_callback(self):
        import tensorflow as tf

        kp = self

        class _KProfCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                kp.begin(epoch)

            def on_test_begin(self, logs=None):
                kp.end()

            def on_epoch_end(self, epoch, logs=None):
                kp.end()
                if kp.last_summary and kp.last_summary.get("epoch") == epoch and logs is not None:
                    logs["train_busy_time_s_cupti"] = kp.last_summary["busy_time_s"]
                    logs["train_cpu_wait_time_s_cupti"] = kp.last_summary["cpu_wait_time_s"]

            def _implements_train_batch_hooks(self):
                return False

            def _implements_test_batch_hooks(self):
                return False

            def _implements_predict_batch_hooks(self):
                return False

        return _KProfCallback()


if __name__ == "__main__":
    assert union_duration([(0, 10), (5, 15)]) == 15
    assert union_duration([(0, 10), (20, 30)]) == 20
    assert union_duration([(0, 10), (10, 20)]) == 20
    assert union_duration([(0, 10), (2, 5)]) == 10
    assert union_duration([]) == 0.0
    assert union_gaps([(0, 10), (20, 30)], 0, 30) == [(10, 20)]
    assert union_gaps([(5, 10)], 0, 20) == [(0, 5), (10, 20)]
    s = summarize([(0, 1e6, "k", 7), (5e5, 1.5e6, "k2", 8)], [], wall_time_s=2.0)
    assert abs(s["busy_time_s"] - 1.5) < 1e-9
    assert abs(s["kernel_sum_time_s"] - 2.0) < 1e-9
    assert abs(s["cpu_wait_time_s"] - 0.5) < 1e-9
    print("gpu_kernel_profile: testes de união OK")
