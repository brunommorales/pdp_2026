from __future__ import annotations

import os
import threading
import time

try:
    import pynvml as _nvml
    _NVML_IMPORTED = True
except Exception:
    _nvml = None
    _NVML_IMPORTED = False

_INIT_DONE = False
_INIT_OK = False


def _ensure_init() -> bool:
    global _INIT_DONE, _INIT_OK
    if _INIT_DONE:
        return _INIT_OK
    _INIT_DONE = True
    if not _NVML_IMPORTED:
        return False
    try:
        _nvml.nvmlInit()
        _INIT_OK = True
    except Exception:
        _INIT_OK = False
    return _INIT_OK


def _resolve_handle(torch_device=None):
    if not _ensure_init():
        return None

    try:
        import torch
        if torch.cuda.is_available():
            idx = torch_device if isinstance(torch_device, int) else torch.cuda.current_device()
            try:
                uuid = torch.cuda.get_device_properties(idx).uuid
                return _nvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode())
            except Exception:
                pass
    except Exception:
        pass

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = cvd.split(",")[0].strip() if cvd else ""
    if first and first not in ("-1",):
        try:
            if first.startswith("GPU-"):
                return _nvml.nvmlDeviceGetHandleByUUID(first.encode())
            if "-" in first:
                return _nvml.nvmlDeviceGetHandleByUUID(f"GPU-{first}".encode())
            return _nvml.nvmlDeviceGetHandleByIndex(int(first))
        except Exception:
            pass

    try:
        return _nvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def _resolve_all_local_handles():
    if not _ensure_init():
        return []
    handles = []
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    try:
        if cvd and cvd not in ("-1",):
            for tok in (t.strip() for t in cvd.split(",") if t.strip()):
                try:
                    if tok.startswith("GPU-"):
                        handles.append(_nvml.nvmlDeviceGetHandleByUUID(tok.encode()))
                    elif "-" in tok:
                        handles.append(_nvml.nvmlDeviceGetHandleByUUID(f"GPU-{tok}".encode()))
                    else:
                        handles.append(_nvml.nvmlDeviceGetHandleByIndex(int(tok)))
                except Exception:
                    pass
        else:
            for i in range(_nvml.nvmlDeviceGetCount()):
                handles.append(_nvml.nvmlDeviceGetHandleByIndex(i))
    except Exception:
        pass
    return handles


def _sum_local_gpus_enabled() -> bool:
    return os.environ.get("HCPA_ENERGY_ALL_VISIBLE_GPUS", "0").strip().lower() in ("1", "true", "yes")


class GpuTelemetry:

    def __init__(self, torch_device=None):
        if _sum_local_gpus_enabled():
            self._handles = _resolve_all_local_handles() or [_resolve_handle(torch_device)]
        else:
            self._handles = [_resolve_handle(torch_device)]
        self._handles = [h for h in self._handles if h is not None]
        self._h = self._handles[0] if self._handles else None

    @property
    def available(self) -> bool:
        return self._h is not None

    @property
    def n_gpus(self) -> int:
        return len(self._handles)

    def energy_j(self):
        if not self._handles:
            return None
        try:
            total = 0.0
            for h in self._handles:
                total += _nvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
            return total
        except Exception:
            return None

    def power_w(self):
        if not self._handles:
            return None
        try:
            total = 0.0
            for h in self._handles:
                total += _nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            return total
        except Exception:
            return None

    def util_pct(self):
        if self._h is None:
            return None
        try:
            if _media_util_todas_gpus() and len(self._handles) > 1:
                v = [float(_nvml.nvmlDeviceGetUtilizationRates(h).gpu) for h in self._handles]
                return sum(v) / len(v)
            return float(_nvml.nvmlDeviceGetUtilizationRates(self._h).gpu)
        except Exception:
            return None

    def mem_util_pct(self):
        if self._h is None:
            return None
        try:
            return float(_nvml.nvmlDeviceGetUtilizationRates(self._h).memory)
        except Exception:
            return None

    def memory_used_mb(self):
        if self._h is None:
            return None
        try:
            return _nvml.nvmlDeviceGetMemoryInfo(self._h).used / (1024.0 ** 2)
        except Exception:
            return None

    def process_memory_used_mb(self, pid=None):
        if self._h is None:
            return None
        pid = pid or os.getpid()
        try:
            for p in _nvml.nvmlDeviceGetComputeRunningProcesses(self._h):
                if p.pid == pid and getattr(p, "usedGpuMemory", None):
                    return p.usedGpuMemory / (1024.0 ** 2)
        except Exception:
            pass
        return None


SAMPLING_DISABLED = -1.0

_SYNC_WARNED = False

DEFAULT_SAMPLE_INTERVAL_S = 0.2


def _effective_interval(passed: float) -> float:
    ov = os.environ.get("HCPA_GPU_SAMPLE_MS")
    if ov:
        try:
            v = float(ov)
            if v == 0:
                return SAMPLING_DISABLED
            if v > 0:
                return v / 1000.0
        except Exception:
            pass
    return passed


def _media_util_todas_gpus() -> bool:
    return os.environ.get("HCPA_UTIL_ALL_GPUS", "0").strip().lower() in ("1", "true", "yes")


def _sidecar_path():
    base = os.environ.get("HCPA_ENERGY_SIDECAR", "").strip()
    if not base:
        return None
    for var in ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        v = os.environ.get(var)
        if v not in (None, ""):
            try:
                if int(v) != 0:
                    return None
            except ValueError:
                pass
            break
    nid = os.environ.get("HCPA_NODE_ID") or os.environ.get("SLURM_NODEID") or "0"
    return f"{base}.node{nid}"


def _sidecar_registra(scope):
    p = _sidecar_path()
    if not p:
        return
    try:
        with open(p, "a") as fh:
            fh.write("{:.6f},{:.6f},{},{},{},{}\n".format(
                scope._t0_wall or 0.0,
                scope.elapsed_s or 0.0,
                "" if scope.energy_j is None else repr(scope.energy_j),
                "" if scope.avg_power_w is None else repr(scope.avg_power_w),
                "" if scope.peak_mem_mb is None else repr(scope.peak_mem_mb),
                scope.tele.n_gpus,
            ))
    except Exception:
        pass


class _Sampler(threading.Thread):

    def __init__(self, tele: GpuTelemetry, interval: float = 0.2):
        super().__init__(daemon=True)
        self._tele = tele
        self._interval = _effective_interval(interval)
        self._stop_event = threading.Event()
        self.mem_samples = []
        self.power_samples = []
        self.util_samples = []
        self.mem_util_samples = []
        self.nvml_calls = 0

    def run(self):
        while not self._stop_event.is_set():
            m = self._tele.memory_used_mb()
            if m is not None:
                self.mem_samples.append(m)
            p = self._tele.power_w()
            if p is not None:
                self.power_samples.append(p)
            u = self._tele.util_pct()
            if u is not None:
                self.util_samples.append(u)
            mu = self._tele.mem_util_pct()
            if mu is not None:
                self.mem_util_samples.append(mu)
            self.nvml_calls += 4
            self._stop_event.wait(self._interval)

    def stop(self):
        self._stop_event.set()


def _cuda_sync():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            return
    except Exception:
        pass

    import sys
    tf = sys.modules.get("tensorflow")
    if tf is None:
        return
    try:
        tf.test.experimental.sync_devices()
    except Exception as exc:
        global _SYNC_WARNED
        if not _SYNC_WARNED:
            _SYNC_WARNED = True
            print("[gpu_energy][ATENCAO] tf.test.experimental.sync_devices() indisponivel "
                  f"({exc}). As bordas de medicao do TensorFlow NAO estao sincronizadas: "
                  "tempo e energia por epoca podem estar deslocados. Verifique a versao do TF.")


class EnergyScope:

    def __init__(self, telemetry: GpuTelemetry, sample_interval: float = 0.2,
                 background_sampling: bool = True):
        self.tele = telemetry
        self._interval = sample_interval
        self._bg = background_sampling
        self.energy_j = None
        self.peak_mem_mb = None
        self.avg_mem_mb = None
        self.avg_power_w = None
        self.peak_power_w = None
        self.elapsed_s = None
        self.avg_util_pct = None
        self.peak_util_pct = None
        self.avg_mem_util_pct = None
        self.peak_mem_util_pct = None
        self.busy_time_s = None
        self.nvml_calls = 0
        self._e0 = None
        self._t0 = None
        self._t0_wall = None
        self._sampler = None
        self._stopped = False

    def start(self):
        return self.__enter__()

    def stop(self):
        return self.__exit__(None, None, None)

    def __enter__(self):
        _cuda_sync()
        self._t0 = time.perf_counter()
        self._t0_wall = time.time()
        self._e0 = self.tele.energy_j()
        if self._bg and self.tele.available \
                and _effective_interval(self._interval) != SAMPLING_DISABLED:
            self._sampler = _Sampler(self.tele, self._interval)
            self._sampler.start()
        return self

    def __exit__(self, *exc):
        if self._stopped:
            return False
        self._stopped = True
        _cuda_sync()
        self.elapsed_s = time.perf_counter() - self._t0
        e1 = self.tele.energy_j()
        if self._e0 is not None and e1 is not None:
            self.energy_j = max(0.0, e1 - self._e0)
        if self._sampler is not None:
            self._sampler.stop()
            self._sampler.join(timeout=2.0)
            self.nvml_calls = self._sampler.nvml_calls
            if self._sampler.mem_samples:
                self.peak_mem_mb = max(self._sampler.mem_samples)
                self.avg_mem_mb = sum(self._sampler.mem_samples) / len(self._sampler.mem_samples)
            if self._sampler.power_samples:
                self.peak_power_w = max(self._sampler.power_samples)
                self.avg_power_w = sum(self._sampler.power_samples) / len(self._sampler.power_samples)
            if self._sampler.util_samples:
                self.peak_util_pct = max(self._sampler.util_samples)
                self.avg_util_pct = sum(self._sampler.util_samples) / len(self._sampler.util_samples)
            if self._sampler.mem_util_samples:
                self.peak_mem_util_pct = max(self._sampler.mem_util_samples)
                self.avg_mem_util_pct = sum(self._sampler.mem_util_samples) / len(self._sampler.mem_util_samples)
        if self.energy_j is None and self.avg_power_w is not None and self.elapsed_s:
            self.energy_j = self.avg_power_w * self.elapsed_s
        if self.avg_util_pct is not None and self.elapsed_s is not None:
            self.busy_time_s = self.elapsed_s * (self.avg_util_pct / 100.0)
        _sidecar_registra(self)
        return False


def make_keras_energy_callback(telemetry: GpuTelemetry | None = None,
                               sample_interval: float = 0.2):
    import tensorflow as tf

    tele = telemetry or GpuTelemetry()

    class _EnergyCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.train_energy_j = None
            self.train_peak_mem_mb = None
            self.train_avg_mem_mb = None
            self.train_avg_power_w = None
            self.train_peak_power_w = None
            self.train_avg_util_pct = None
            self.train_peak_util_pct = None
            self.train_busy_time_s = None
            self._scope = None

        def on_train_begin(self, logs=None):
            self._scope = EnergyScope(tele, sample_interval)
            self._scope.__enter__()

        def on_train_end(self, logs=None):
            if self._scope is not None:
                self._scope.__exit__(None, None, None)
                self.train_energy_j = self._scope.energy_j
                self.train_peak_mem_mb = self._scope.peak_mem_mb
                self.train_avg_mem_mb = self._scope.avg_mem_mb
                self.train_avg_power_w = self._scope.avg_power_w
                self.train_peak_power_w = self._scope.peak_power_w
                self.train_avg_util_pct = self._scope.avg_util_pct
                self.train_peak_util_pct = self._scope.peak_util_pct
                self.train_busy_time_s = self._scope.busy_time_s

    return _EnergyCallback()
