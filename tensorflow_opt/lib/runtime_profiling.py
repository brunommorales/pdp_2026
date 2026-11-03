from __future__ import annotations

import atexit
import contextlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import tensorflow as tf


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "trace"


@dataclass
class RuntimeProfilingConfig:
    enable_tf_profiler: bool
    rank0_only: bool
    wait_steps: int
    warmup_steps: int
    active_steps: int
    repeat: int
    max_steps: int
    stages: tuple[str, ...]
    output_dir: str
    host_tracer_level: int
    python_tracer_level: int
    device_tracer_level: int

    @classmethod
    def from_env(cls, output_dir: Path) -> "RuntimeProfilingConfig":
        stages_raw = os.environ.get("HCPA_PROFILE_STAGES", "train,val,test,eval")
        stages = tuple(stage.strip() for stage in stages_raw.split(",") if stage.strip())
        requested_dir = (
            os.environ.get("HCPA_RUNTIME_TRACE_DIR")
            or os.environ.get("HCPA_TORCH_PROFILER_DIR")
            or str(output_dir)
        )
        return cls(
            enable_tf_profiler=_env_flag(
                "HCPA_ENABLE_TF_PROFILER",
                _env_flag(
                    "HCPA_ENABLE_RUNTIME_TRACE",
                    _env_flag("HCPA_ENABLE_TORCH_PROFILER", False),
                ),
            ),
            rank0_only=_env_flag("HCPA_PROFILE_RANK0_ONLY", True),
            wait_steps=max(0, _env_int("HCPA_TORCH_PROFILER_WAIT", 1)),
            warmup_steps=max(0, _env_int("HCPA_TORCH_PROFILER_WARMUP", 1)),
            active_steps=max(1, _env_int("HCPA_TORCH_PROFILER_ACTIVE", 3)),
            repeat=max(1, _env_int("HCPA_TORCH_PROFILER_REPEAT", 1)),
            max_steps=max(0, _env_int("HCPA_TORCH_PROFILER_MAX_STEPS", 0)),
            stages=stages,
            output_dir=str(Path(requested_dir)),
            host_tracer_level=max(0, _env_int("HCPA_TF_HOST_TRACER_LEVEL", 2)),
            python_tracer_level=max(0, _env_int("HCPA_TF_PYTHON_TRACER_LEVEL", 1)),
            device_tracer_level=max(0, _env_int("HCPA_TF_DEVICE_TRACER_LEVEL", 1)),
        )


class _FitProfilerCallback(tf.keras.callbacks.Callback):
    def __init__(self, runtime_profiler: "RuntimeProfiler") -> None:
        super().__init__()
        self.runtime_profiler = runtime_profiler
        self._train_batch_profiled = False
        self._test_batch_profiled = False

    def on_train_begin(self, logs=None):
        self.runtime_profiler.on_train_begin()

    def on_train_end(self, logs=None):
        self.runtime_profiler.on_train_end()

    def on_train_batch_begin(self, batch, logs=None):
        self._train_batch_profiled = self.runtime_profiler.on_train_batch_begin()

    def on_train_batch_end(self, batch, logs=None):
        self.runtime_profiler.on_train_batch_end(self._train_batch_profiled)
        self._train_batch_profiled = False

    def on_test_batch_begin(self, batch, logs=None):
        self._test_batch_profiled = self.runtime_profiler.on_test_batch_begin()

    def on_test_batch_end(self, batch, logs=None):
        self.runtime_profiler.on_test_batch_end(self._test_batch_profiled)
        self._test_batch_profiled = False

    def _implements_train_batch_hooks(self):
        return self.runtime_profiler.stage_enabled("train")

    def _implements_test_batch_hooks(self):
        return self.runtime_profiler.stage_enabled("val")

    def _implements_predict_batch_hooks(self):
        return False


class RuntimeProfiler:

    def __init__(
        self,
        output_root: str | os.PathLike[str],
        *,
        rank: int = 0,
        project_name: str = "tensorflow_opt",
    ) -> None:
        self.rank = int(rank)
        self.project_name = project_name
        requested_dir = os.environ.get("HCPA_RUNTIME_TRACE_DIR") or os.environ.get("HCPA_TORCH_PROFILER_DIR")
        self.output_dir = Path(requested_dir) if requested_dir else Path(output_root) / "profiling" / "tf_traces"
        self.config = RuntimeProfilingConfig.from_env(self.output_dir)
        self.active = (not self.config.rank0_only) or self.rank == 0
        self.enabled = self.active and self.config.enable_tf_profiler
        self.train_step_count = 0
        self.val_step_count = 0
        self.profile_index = 0
        self.fit_profile_dirs: list[str] = []
        self.manual_profile_dirs: list[str] = []
        self._fit_callback = _FitProfilerCallback(self) if self.enabled else None
        self._current_profile_dir: Optional[Path] = None
        self._current_trace_cm = None
        self._session_active = False
        self._closed = False

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        atexit.register(self.close)

    def stage_enabled(self, stage: str) -> bool:
        return self.enabled and stage in self.config.stages

    def fit_callback(self) -> Optional[tf.keras.callbacks.Callback]:
        if not self.enabled:
            return None
        if not any(stage in self.config.stages for stage in ("train", "val")):
            return None
        return self._fit_callback

    def _profiler_options(self):
        options_cls = getattr(tf.profiler.experimental, "ProfilerOptions", None)
        if options_cls is None:
            return None
        return options_cls(
            host_tracer_level=self.config.host_tracer_level,
            python_tracer_level=self.config.python_tracer_level,
            device_tracer_level=self.config.device_tracer_level,
        )

    def _schedule_cycle_length(self) -> int:
        return max(1, self.config.wait_steps + self.config.warmup_steps + self.config.active_steps)

    def _should_profile_train_step(self, step_idx: int) -> bool:
        if not self.stage_enabled("train"):
            return False
        if self.config.max_steps > 0 and step_idx >= self.config.max_steps:
            return False
        cycle_length = self._schedule_cycle_length()
        cycle_idx = step_idx // cycle_length
        if cycle_idx >= self.config.repeat:
            return False
        cycle_pos = step_idx % cycle_length
        profile_start = self.config.wait_steps + self.config.warmup_steps
        return profile_start <= cycle_pos < cycle_length

    def _is_last_profiled_train_step(self, step_idx: int) -> bool:
        cycle_length = self._schedule_cycle_length()
        cycle_pos = step_idx % cycle_length
        if self.config.max_steps > 0 and (step_idx + 1) >= self.config.max_steps:
            return True
        return cycle_pos == (cycle_length - 1)

    def _start_session(self, label: str) -> bool:
        if self._session_active:
            return True
        profile_dir = self.output_dir / f"{self.project_name}_rank{self.rank}_{_sanitize_name(label)}_{self.profile_index:03d}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            options = self._profiler_options()
            if options is None:
                tf.profiler.experimental.start(str(profile_dir))
            else:
                tf.profiler.experimental.start(str(profile_dir), options=options)
        except Exception as exc:
            print(f"[profiling] falha ao iniciar tf.profiler em {profile_dir}: {exc}")
            return False
        self._session_active = True
        self._current_profile_dir = profile_dir
        self.fit_profile_dirs.append(profile_dir.name)
        self.profile_index += 1
        return True

    def _stop_session(self) -> None:
        if not self._session_active:
            return
        try:
            tf.profiler.experimental.stop()
        except Exception:
            pass
        self._session_active = False
        self._current_profile_dir = None

    def _enter_trace(self, name: str, step_num: int) -> None:
        if not self._session_active:
            self._current_trace_cm = None
            return
        try:
            trace_cm = tf.profiler.experimental.Trace(name, step_num=step_num, _r=1)
            trace_cm.__enter__()
            self._current_trace_cm = trace_cm
        except Exception:
            self._current_trace_cm = None

    def _exit_trace(self) -> None:
        if self._current_trace_cm is None:
            return
        try:
            self._current_trace_cm.__exit__(None, None, None)
        except Exception:
            pass
        finally:
            self._current_trace_cm = None

    def on_train_begin(self) -> None:
        self._exit_trace()

    def on_train_end(self) -> None:
        self._exit_trace()
        self._stop_session()

    def on_train_batch_begin(self) -> bool:
        step_idx = self.train_step_count
        should_profile = self._should_profile_train_step(step_idx)
        if should_profile and not self._session_active:
            if not self._start_session("train"):
                return False
        if should_profile:
            self._enter_trace("train/batch", step_idx + 1)
        return should_profile

    def on_train_batch_end(self, profiled: bool) -> None:
        step_idx = self.train_step_count
        if profiled:
            self._exit_trace()
            if self._is_last_profiled_train_step(step_idx):
                self._stop_session()
        self.train_step_count += 1

    def on_test_batch_begin(self) -> bool:
        if not self.stage_enabled("val") or not self._session_active:
            return False
        self._enter_trace("val/batch", self.val_step_count + 1)
        return True

    def on_test_batch_end(self, profiled: bool) -> None:
        if profiled:
            self._exit_trace()
        self.val_step_count += 1

    @contextlib.contextmanager
    def profile_scope(self, name: str, *, stage: str = "eval"):
        if not self.stage_enabled(stage):
            yield
            return

        profile_dir = self.output_dir / f"{self.project_name}_rank{self.rank}_{_sanitize_name(name)}_{self.profile_index:03d}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        started = False
        trace_cm = None
        record_output = False
        try:
            options = self._profiler_options()
            if options is None:
                tf.profiler.experimental.start(str(profile_dir))
            else:
                tf.profiler.experimental.start(str(profile_dir), options=options)
            started = True
            trace_cm = tf.profiler.experimental.Trace(name, step_num=self.profile_index, _r=1)
            trace_cm.__enter__()
        except Exception as exc:
            print(f"[profiling] falha ao perfilar '{name}': {exc}")
        else:
            record_output = True
        try:
            yield
        finally:
            if trace_cm is not None:
                try:
                    trace_cm.__exit__(None, None, None)
                except Exception:
                    pass
            if started:
                try:
                    tf.profiler.experimental.stop()
                except Exception:
                    pass
            if record_output:
                self.manual_profile_dirs.append(profile_dir.name)
                self.profile_index += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._exit_trace()
        self._stop_session()
        if not self.enabled:
            return

        manifest_path = self.output_dir / f"{self.project_name}_rank{self.rank}_profiling_manifest.json"
        payload = {
            "project_name": self.project_name,
            "rank": self.rank,
            "train_step_count": self.train_step_count,
            "val_step_count": self.val_step_count,
            "fit_profile_dirs": self.fit_profile_dirs,
            "manual_profile_dirs": self.manual_profile_dirs,
            "config": asdict(self.config),
        }
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        except Exception:
            pass
