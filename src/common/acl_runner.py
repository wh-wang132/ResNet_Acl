from __future__ import annotations

import ctypes
import itertools
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator

import acl
import numpy as np

from .artifact_scanner import ArtifactRecord, TensorSpec
from .data import PreloadedSample, RuntimeInputAdapter, validate_preloaded_sample_shapes

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0
QUEUE_POLL_TIMEOUT_SECONDS = 0.1
ITEMSIZE_TO_TENSOR_META: dict[int, tuple[np.dtype, int]] = {
    int(np.dtype(np.float16).itemsize): (np.dtype(np.float16), 10),
    int(np.dtype(np.float32).itemsize): (np.dtype(np.float32), 1),
}


class ACLRuntimeError(RuntimeError):
    """ACL 推理运行时初始化或执行失败。"""


class _ACLRuntimeManager:
    _lock = threading.Lock()
    _ref_count = 0
    _device_id: int | None = None

    @classmethod
    def acquire(cls, device_id: int) -> None:
        with cls._lock:
            if cls._ref_count == 0:
                _check_ret(acl.init(), "acl.init")
                _check_ret(acl.rt.set_device(device_id), f"acl.rt.set_device({device_id})")
                cls._device_id = int(device_id)
            elif cls._device_id != int(device_id):
                raise ACLRuntimeError(
                    f"当前进程已绑定 device_id={cls._device_id}，不能并发切换到 device_id={device_id}"
                )
            cls._ref_count += 1

    @classmethod
    def release(cls, device_id: int) -> None:
        with cls._lock:
            if cls._ref_count <= 0:
                return
            cls._ref_count -= 1
            if cls._ref_count > 0:
                return
            _check_ret(acl.rt.reset_device(device_id), f"acl.rt.reset_device({device_id})", raise_on_error=False)
            _check_ret(acl.finalize(), "acl.finalize", raise_on_error=False)
            cls._device_id = None


def _unwrap_with_ret(result: Any, action: str) -> Any:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ACLRuntimeError(f"{action} 返回值格式异常: {result!r}")
    value, ret = result
    if ret != 0:
        raise ACLRuntimeError(f"{action} 失败，ret={ret}")
    return value


def _check_ret(result: int, action: str, *, raise_on_error: bool = True) -> bool:
    if result == 0:
        return True
    if raise_on_error:
        raise ACLRuntimeError(f"{action} 失败，ret={result}")
    return False


@dataclass(frozen=True)
class OrderedInferenceResult:
    seq_id: int
    sample_index: int
    preloaded: PreloadedSample
    outputs: tuple[np.ndarray, ...]
    pure_infer_ms: float
    dispatch_started_ns: int
    worker_id: int


@dataclass(frozen=True)
class _InferenceTask:
    seq_id: int
    sample_index: int
    preloaded: PreloadedSample
    dispatch_started_ns: int


@dataclass(frozen=True)
class _WorkerReady:
    worker_id: int
    input_spec: TensorSpec
    output_specs: tuple[TensorSpec, ...]


@dataclass(frozen=True)
class _WorkerWarmed:
    worker_id: int


@dataclass(frozen=True)
class _WorkerFailure:
    worker_id: int
    phase: str
    error: Exception


@dataclass
class _WorkerState:
    worker_id: int
    input_queue: queue.Queue[_InferenceTask]
    output_queue: queue.Queue[OrderedInferenceResult]
    thread: threading.Thread


class ACLModelRunner:
    def __init__(self, artifact: ArtifactRecord, device_id: int = 0, *, use_async_stream: bool = False):
        self.artifact = artifact
        self.device_id = int(device_id)
        self._use_async_stream = bool(use_async_stream)
        self._context: int | None = None
        self._stream: int | None = None
        self._model_id: int | None = None
        self._model_desc: int | None = None
        self._input_dataset: int | None = None
        self._output_dataset: int | None = None
        self._input_buffers: list[int] = []
        self._output_buffers: list[int] = []
        self._input_device_ptrs: list[int] = []
        self._output_device_ptrs: list[int] = []
        self._input_sizes: list[int] = []
        self._output_sizes: list[int] = []
        self._input_host_ptr: int | None = None
        self._output_host_ptrs: list[int] = []
        self._model_input_spec: TensorSpec | None = None
        self._model_output_specs: tuple[TensorSpec, ...] = ()
        self._runtime_acquired = False
        self._opened = False

    def __enter__(self) -> "ACLModelRunner":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._opened:
            return
        _ACLRuntimeManager.acquire(self.device_id)
        self._runtime_acquired = True
        try:
            self._context = _unwrap_with_ret(acl.rt.create_context(self.device_id), "acl.rt.create_context")
            _check_ret(acl.rt.set_context(self._context), "acl.rt.set_context")
            if self._use_async_stream:
                self._stream = _unwrap_with_ret(acl.rt.create_stream(), "acl.rt.create_stream")
            self._model_id = _unwrap_with_ret(
                acl.mdl.load_from_file(str(self.artifact.model_path)),
                f"acl.mdl.load_from_file({self.artifact.model_path})",
            )
            self._model_desc = acl.mdl.create_desc()
            _check_ret(acl.mdl.get_desc(self._model_desc, self._model_id), "acl.mdl.get_desc")
            self._load_model_specs()
            self._prepare_io()
            self._opened = True
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._input_dataset is not None:
            self._destroy_dataset(self._input_dataset, self._input_buffers)
            self._input_dataset = None
        if self._output_dataset is not None:
            self._destroy_dataset(self._output_dataset, self._output_buffers)
            self._output_dataset = None
        for ptr in self._input_device_ptrs:
            _check_ret(acl.rt.free(ptr), f"acl.rt.free(input_ptr={ptr})", raise_on_error=False)
        self._input_device_ptrs.clear()
        for ptr in self._output_device_ptrs:
            _check_ret(acl.rt.free(ptr), f"acl.rt.free(output_ptr={ptr})", raise_on_error=False)
        self._output_device_ptrs.clear()
        self._input_sizes.clear()
        self._output_sizes.clear()
        if self._input_host_ptr is not None:
            _check_ret(acl.rt.free_host(self._input_host_ptr), "acl.rt.free_host(input_host_ptr)", raise_on_error=False)
            self._input_host_ptr = None
        for ptr in self._output_host_ptrs:
            _check_ret(acl.rt.free_host(ptr), f"acl.rt.free_host(output_host_ptr={ptr})", raise_on_error=False)
        self._output_host_ptrs.clear()
        if self._model_desc is not None:
            _check_ret(acl.mdl.destroy_desc(self._model_desc), "acl.mdl.destroy_desc", raise_on_error=False)
            self._model_desc = None
        if self._model_id is not None:
            _check_ret(acl.mdl.unload(self._model_id), "acl.mdl.unload", raise_on_error=False)
            self._model_id = None
        if self._stream is not None:
            _check_ret(acl.rt.destroy_stream(self._stream), "acl.rt.destroy_stream", raise_on_error=False)
            self._stream = None
        if self._context is not None:
            _check_ret(acl.rt.destroy_context(self._context), "acl.rt.destroy_context", raise_on_error=False)
            self._context = None
        if self._runtime_acquired:
            _ACLRuntimeManager.release(self.device_id)
            self._runtime_acquired = False
        self._model_input_spec = None
        self._model_output_specs = ()
        self._opened = False

    def infer(self, input_array: np.ndarray) -> tuple[np.ndarray, ...]:
        outputs, _ = self.infer_with_timing(input_array)
        return outputs

    def infer_with_timing(self, input_array: np.ndarray) -> tuple[tuple[np.ndarray, ...], float]:
        self._ensure_opened()
        if self._use_async_stream:
            return self._infer_with_async_stream(input_array)
        return self._infer_with_sync(input_array)

    @property
    def input_spec(self) -> TensorSpec:
        if self._model_input_spec is None:
            raise ACLRuntimeError("ACLModelRunner 尚未读取模型输入规格")
        return self._model_input_spec

    @property
    def output_specs(self) -> tuple[TensorSpec, ...]:
        if not self._model_output_specs:
            raise ACLRuntimeError("ACLModelRunner 尚未读取模型输出规格")
        return self._model_output_specs

    def _infer_with_sync(self, input_array: np.ndarray) -> tuple[tuple[np.ndarray, ...], float]:
        _check_ret(acl.rt.set_context(self._context), "acl.rt.set_context")
        self._validate_input_array(input_array, self.input_spec)
        host_input = input_array.tobytes()
        _check_ret(
            acl.rt.memcpy(
                self._input_device_ptrs[0],
                self._input_sizes[0],
                acl.util.bytes_to_ptr(host_input),
                len(host_input),
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "acl.rt.memcpy(host_to_device)",
        )
        start = time.perf_counter()
        _check_ret(
            acl.mdl.execute(self._model_id, self._input_dataset, self._output_dataset),
            "acl.mdl.execute",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        outputs: list[np.ndarray] = []
        for index, spec in enumerate(self.output_specs):
            _check_ret(
                acl.rt.memcpy(
                    self._output_host_ptrs[index],
                    self._output_sizes[index],
                    self._output_device_ptrs[index],
                    self._output_sizes[index],
                    ACL_MEMCPY_DEVICE_TO_HOST,
                ),
                f"acl.rt.memcpy(device_to_host, output_index={index})",
            )
            outputs.append(self._read_output_array(index, spec))
        return tuple(outputs), elapsed_ms

    def _infer_with_async_stream(self, input_array: np.ndarray) -> tuple[tuple[np.ndarray, ...], float]:
        if self._stream is None or self._input_host_ptr is None:
            raise ACLRuntimeError("异步推理模式缺少 stream 或 host staging buffer")
        _check_ret(acl.rt.set_context(self._context), "acl.rt.set_context")
        self._validate_input_array(input_array, self.input_spec)

        ctypes.memmove(self._input_host_ptr, acl.util.numpy_to_ptr(input_array), self._input_sizes[0])
        _check_ret(
            acl.rt.memcpy_async(
                self._input_device_ptrs[0],
                self._input_sizes[0],
                self._input_host_ptr,
                self._input_sizes[0],
                ACL_MEMCPY_HOST_TO_DEVICE,
                self._stream,
            ),
            "acl.rt.memcpy_async(host_to_device)",
        )
        _check_ret(acl.rt.synchronize_stream(self._stream), "acl.rt.synchronize_stream(host_to_device)")

        start = time.perf_counter()
        _check_ret(
            acl.mdl.execute_async(self._model_id, self._input_dataset, self._output_dataset, self._stream),
            "acl.mdl.execute_async",
        )
        _check_ret(acl.rt.synchronize_stream(self._stream), "acl.rt.synchronize_stream(execute_async)")
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        for index in range(len(self.output_specs)):
            _check_ret(
                acl.rt.memcpy_async(
                    self._output_host_ptrs[index],
                    self._output_sizes[index],
                    self._output_device_ptrs[index],
                    self._output_sizes[index],
                    ACL_MEMCPY_DEVICE_TO_HOST,
                    self._stream,
                ),
                f"acl.rt.memcpy_async(device_to_host, output_index={index})",
            )
        _check_ret(acl.rt.synchronize_stream(self._stream), "acl.rt.synchronize_stream(device_to_host)")

        outputs = [self._read_output_array(index, spec) for index, spec in enumerate(self.output_specs)]
        return tuple(outputs), elapsed_ms

    def _read_output_array(self, index: int, spec: TensorSpec) -> np.ndarray:
        output_bytes = acl.util.ptr_to_bytes(self._output_host_ptrs[index], self._output_sizes[index])
        return np.frombuffer(output_bytes, dtype=spec.dtype).reshape(spec.shape).copy()

    def _load_model_specs(self) -> None:
        input_count = int(acl.mdl.get_num_inputs(self._model_desc))
        output_count = int(acl.mdl.get_num_outputs(self._model_desc))
        if input_count != 1:
            raise ACLRuntimeError(f"当前仅支持单输入 OM，实际输入数为 {input_count}: {self.artifact.model_path}")
        if output_count != len(self.artifact.output_specs):
            raise ACLRuntimeError(
                f"OM 输出数与摘要不一致: model={output_count}, summary={len(self.artifact.output_specs)}"
            )

        input_shape = self._read_tensor_shape(acl.mdl.get_input_dims(self._model_desc, 0), "input")
        input_size = int(acl.mdl.get_input_size_by_index(self._model_desc, 0))
        input_dtype, input_elem_type = self._infer_tensor_meta(input_shape, input_size, "input")
        self._model_input_spec = TensorSpec(
            name=self.artifact.input_spec.name,
            shape=input_shape,
            dtype=input_dtype,
            elem_type=input_elem_type,
        )

        output_specs: list[TensorSpec] = []
        for index, artifact_output_spec in enumerate(self.artifact.output_specs):
            output_shape = self._read_tensor_shape(acl.mdl.get_output_dims(self._model_desc, index), f"output[{index}]")
            output_size = int(acl.mdl.get_output_size_by_index(self._model_desc, index))
            output_dtype, output_elem_type = self._infer_tensor_meta(output_shape, output_size, f"output[{index}]")
            output_specs.append(
                TensorSpec(
                    name=artifact_output_spec.name,
                    shape=output_shape,
                    dtype=output_dtype,
                    elem_type=output_elem_type,
                )
            )
        self._model_output_specs = tuple(output_specs)

    def _prepare_io(self) -> None:
        self._input_dataset = acl.mdl.create_dataset()
        self._output_dataset = acl.mdl.create_dataset()

        input_size = int(acl.mdl.get_input_size_by_index(self._model_desc, 0))
        expected_input_size = self._expected_nbytes(self.input_spec)
        if input_size != expected_input_size:
            raise ACLRuntimeError(
                f"OM 输入字节数与摘要不一致: model={input_size}, summary={expected_input_size}"
            )
        input_ptr = _unwrap_with_ret(
            acl.rt.malloc(input_size, ACL_MEM_MALLOC_HUGE_FIRST),
            f"acl.rt.malloc(input_size={input_size})",
        )
        input_buffer = acl.create_data_buffer(input_ptr, input_size)
        self._add_dataset_buffer(self._input_dataset, input_buffer, "input")
        self._input_device_ptrs.append(input_ptr)
        self._input_buffers.append(input_buffer)
        self._input_sizes.append(input_size)
        if self._use_async_stream:
            self._input_host_ptr = _unwrap_with_ret(
                acl.rt.malloc_host(input_size),
                f"acl.rt.malloc_host(input_size={input_size})",
            )

        for index, spec in enumerate(self.output_specs):
            output_size = int(acl.mdl.get_output_size_by_index(self._model_desc, index))
            expected_output_size = self._expected_nbytes(spec)
            if output_size != expected_output_size:
                raise ACLRuntimeError(
                    f"OM 输出字节数与摘要不一致: output_index={index}, model={output_size}, summary={expected_output_size}"
                )
            output_ptr = _unwrap_with_ret(
                acl.rt.malloc(output_size, ACL_MEM_MALLOC_HUGE_FIRST),
                f"acl.rt.malloc(output_size={output_size})",
            )
            output_host_ptr = _unwrap_with_ret(
                acl.rt.malloc_host(output_size),
                f"acl.rt.malloc_host(output_size={output_size})",
            )
            output_buffer = acl.create_data_buffer(output_ptr, output_size)
            self._add_dataset_buffer(self._output_dataset, output_buffer, f"output[{index}]")
            self._output_device_ptrs.append(output_ptr)
            self._output_host_ptrs.append(output_host_ptr)
            self._output_buffers.append(output_buffer)
            self._output_sizes.append(output_size)

    def _destroy_dataset(self, dataset: int, buffers: list[int]) -> None:
        for buffer in buffers:
            _check_ret(acl.destroy_data_buffer(buffer), "acl.destroy_data_buffer", raise_on_error=False)
        buffers.clear()
        _check_ret(acl.mdl.destroy_dataset(dataset), "acl.mdl.destroy_dataset", raise_on_error=False)

    @staticmethod
    def _validate_input_array(input_array: np.ndarray, spec: TensorSpec) -> None:
        if tuple(input_array.shape) != spec.shape:
            raise ACLRuntimeError(
                f"输入 shape 不匹配: expected={spec.shape}, actual={tuple(input_array.shape)}"
            )
        if np.dtype(input_array.dtype) != np.dtype(spec.dtype):
            raise ACLRuntimeError(f"输入 dtype 不匹配: expected={spec.dtype}, actual={input_array.dtype}")
        if not input_array.flags.c_contiguous:
            raise ACLRuntimeError("输入数组必须是 C contiguous")

    @staticmethod
    def _read_tensor_shape(raw_dims: tuple[dict[str, Any], int], name: str) -> tuple[int, ...]:
        dims_info, ret = raw_dims
        if ret != 0:
            raise ACLRuntimeError(f"读取 {name} 维度失败，ret={ret}")
        return tuple(int(dim) for dim in dims_info["dims"][: dims_info["dimCount"]])

    @staticmethod
    def _infer_tensor_meta(shape: tuple[int, ...], buffer_size: int, name: str) -> tuple[np.dtype, int]:
        element_count = int(np.prod(shape))
        if element_count <= 0:
            raise ACLRuntimeError(f"{name} shape 非法，无法推断 dtype: shape={shape}")
        if buffer_size % element_count != 0:
            raise ACLRuntimeError(
                f"{name} 字节数无法被 shape 整除，无法推断 dtype: shape={shape}, buffer_size={buffer_size}"
            )
        itemsize = buffer_size // element_count
        try:
            return ITEMSIZE_TO_TENSOR_META[itemsize]
        except KeyError as exc:
            raise ACLRuntimeError(
                f"{name} 不支持的元素字节数，无法推断 dtype: shape={shape}, buffer_size={buffer_size}"
            ) from exc

    @staticmethod
    def _expected_nbytes(spec: TensorSpec) -> int:
        return int(np.prod(spec.shape)) * int(np.dtype(spec.dtype).itemsize)

    @staticmethod
    def _add_dataset_buffer(dataset: int, buffer: int, name: str) -> None:
        _, ret = ACLModelRunner._normalize_add_buffer_result(
            acl.mdl.add_dataset_buffer(dataset, buffer),
            name,
        )
        if ret != 0:
            raise ACLRuntimeError(f"acl.mdl.add_dataset_buffer({name}) 失败，ret={ret}")

    @staticmethod
    def _normalize_add_buffer_result(result: Any, name: str) -> tuple[Any, int]:
        if not isinstance(result, tuple) or len(result) != 2:
            raise ACLRuntimeError(f"acl.mdl.add_dataset_buffer({name}) 返回值格式异常: {result!r}")
        return result

    def _ensure_opened(self) -> None:
        if not self._opened or self._context is None or self._model_id is None:
            raise ACLRuntimeError("ACLModelRunner 尚未 open")


class ACLConcurrentOrderedExecutor:
    def __init__(self, artifact: ArtifactRecord, *, device_id: int = 0, num_instances: int = 1, buffer_depth: int = 1):
        self.artifact = artifact
        self.device_id = int(device_id)
        self.num_instances = int(num_instances)
        self.buffer_depth = int(buffer_depth)
        self._active_instances = 1
        self._model_input_spec: TensorSpec | None = None
        self._model_output_specs: tuple[TensorSpec, ...] = ()
        self._run_started_ns: int | None = None
        self._last_result_ready_ns: int | None = None

    @property
    def active_instances(self) -> int:
        return self._active_instances

    @property
    def input_spec(self) -> TensorSpec:
        if self._model_input_spec is None:
            raise ACLRuntimeError("并发执行器尚未读取模型输入规格")
        return self._model_input_spec

    @property
    def output_specs(self) -> tuple[TensorSpec, ...]:
        if not self._model_output_specs:
            raise ACLRuntimeError("并发执行器尚未读取模型输出规格")
        return self._model_output_specs

    @property
    def run_started_ns(self) -> int | None:
        return self._run_started_ns

    @property
    def last_result_ready_ns(self) -> int | None:
        return self._last_result_ready_ns

    def iter_ordered(
        self,
        preloaded_samples: list[PreloadedSample],
        *,
        repeat: int = 1,
        warmup_steps: int = 0,
    ) -> Iterator[OrderedInferenceResult]:
        if repeat <= 0:
            raise ValueError("repeat 必须为正整数")
        if warmup_steps < 0:
            raise ValueError("warmup_steps 不能为负数")
        if not preloaded_samples:
            return

        self._active_instances = min(self.num_instances, len(preloaded_samples))
        self._model_input_spec = None
        self._model_output_specs = ()
        self._run_started_ns = None
        self._last_result_ready_ns = None

        start_event = threading.Event()
        stop_event = threading.Event()
        ready_queue: queue.Queue[_WorkerReady] = queue.Queue()
        warmed_queue: queue.Queue[_WorkerWarmed] = queue.Queue()
        error_queue: queue.Queue[_WorkerFailure] = queue.Queue()
        workers = self._start_workers(
            preloaded_samples,
            warmup_steps=warmup_steps,
            start_event=start_event,
            stop_event=stop_event,
            ready_queue=ready_queue,
            warmed_queue=warmed_queue,
            error_queue=error_queue,
        )
        dispatcher_thread: threading.Thread | None = None

        try:
            self._await_worker_ready(ready_queue, error_queue, stop_event)
            validate_preloaded_sample_shapes(preloaded_samples, self.artifact.input_spec, self.input_spec)
            start_event.set()
            self._await_worker_warmup(warmed_queue, error_queue, stop_event)

            self._run_started_ns = time.perf_counter_ns()
            dispatcher_thread = threading.Thread(
                target=self._dispatcher_main,
                args=(preloaded_samples, repeat, workers, stop_event, error_queue),
                daemon=True,
                name="acl-dispatcher",
            )
            dispatcher_thread.start()

            total_jobs = len(preloaded_samples) * repeat
            for expected_seq in range(total_jobs):
                failure = self._poll_failure(error_queue)
                if failure is not None:
                    raise ACLRuntimeError(
                        f"并发推理在 {failure.phase} 阶段失败，worker_id={failure.worker_id}: {failure.error}"
                    ) from failure.error

                worker_id = expected_seq % self._active_instances
                output_queue = workers[worker_id].output_queue
                while True:
                    try:
                        result = output_queue.get(timeout=QUEUE_POLL_TIMEOUT_SECONDS)
                    except queue.Empty:
                        failure = self._poll_failure(error_queue)
                        if failure is not None:
                            raise ACLRuntimeError(
                                f"并发推理在 {failure.phase} 阶段失败，worker_id={failure.worker_id}: {failure.error}"
                            ) from failure.error
                        continue
                    if result.seq_id != expected_seq:
                        raise ACLRuntimeError(
                            f"并发推理顺序错误: expected_seq={expected_seq}, actual_seq={result.seq_id}, worker_id={worker_id}"
                        )
                    self._last_result_ready_ns = time.perf_counter_ns()
                    yield result
                    break
        finally:
            stop_event.set()
            if dispatcher_thread is not None:
                dispatcher_thread.join()
            for worker in workers:
                worker.thread.join()

    def _start_workers(
        self,
        preloaded_samples: list[PreloadedSample],
        *,
        warmup_steps: int,
        start_event: threading.Event,
        stop_event: threading.Event,
        ready_queue: queue.Queue[_WorkerReady],
        warmed_queue: queue.Queue[_WorkerWarmed],
        error_queue: queue.Queue[_WorkerFailure],
    ) -> list[_WorkerState]:
        workers: list[_WorkerState] = []
        for worker_id in range(self._active_instances):
            input_queue: queue.Queue[_InferenceTask] = queue.Queue(maxsize=self.buffer_depth)
            output_queue: queue.Queue[OrderedInferenceResult] = queue.Queue(maxsize=self.buffer_depth)
            warmup_samples = preloaded_samples[worker_id:: self._active_instances]
            thread = threading.Thread(
                target=self._worker_main,
                args=(
                    worker_id,
                    warmup_samples,
                    warmup_steps,
                    input_queue,
                    output_queue,
                    start_event,
                    stop_event,
                    ready_queue,
                    warmed_queue,
                    error_queue,
                ),
                daemon=True,
                name=f"acl-worker-{worker_id}",
            )
            thread.start()
            workers.append(_WorkerState(worker_id=worker_id, input_queue=input_queue, output_queue=output_queue, thread=thread))
        return workers

    def _await_worker_ready(
        self,
        ready_queue: queue.Queue[_WorkerReady],
        error_queue: queue.Queue[_WorkerFailure],
        stop_event: threading.Event,
    ) -> None:
        ready_count = 0
        while ready_count < self._active_instances:
            failure = self._poll_failure(error_queue)
            if failure is not None:
                stop_event.set()
                raise ACLRuntimeError(
                    f"worker 启动失败，worker_id={failure.worker_id}, phase={failure.phase}: {failure.error}"
                ) from failure.error
            try:
                ready = ready_queue.get(timeout=QUEUE_POLL_TIMEOUT_SECONDS)
            except queue.Empty:
                continue
            if self._model_input_spec is None:
                self._model_input_spec = ready.input_spec
                self._model_output_specs = ready.output_specs
            elif ready.input_spec != self._model_input_spec or ready.output_specs != self._model_output_specs:
                stop_event.set()
                raise ACLRuntimeError(
                    f"并发 worker 加载出的模型规格不一致: worker_id={ready.worker_id}, "
                    f"input_spec={ready.input_spec}, output_specs={ready.output_specs}"
                )
            ready_count += 1

    def _await_worker_warmup(
        self,
        warmed_queue: queue.Queue[_WorkerWarmed],
        error_queue: queue.Queue[_WorkerFailure],
        stop_event: threading.Event,
    ) -> None:
        warmed_count = 0
        while warmed_count < self._active_instances:
            failure = self._poll_failure(error_queue)
            if failure is not None:
                stop_event.set()
                raise ACLRuntimeError(
                    f"worker warmup 失败，worker_id={failure.worker_id}, phase={failure.phase}: {failure.error}"
                ) from failure.error
            try:
                warmed_queue.get(timeout=QUEUE_POLL_TIMEOUT_SECONDS)
            except queue.Empty:
                continue
            warmed_count += 1

    def _worker_main(
        self,
        worker_id: int,
        warmup_samples: list[PreloadedSample],
        warmup_steps: int,
        input_queue: queue.Queue[_InferenceTask],
        output_queue: queue.Queue[OrderedInferenceResult],
        start_event: threading.Event,
        stop_event: threading.Event,
        ready_queue: queue.Queue[_WorkerReady],
        warmed_queue: queue.Queue[_WorkerWarmed],
        error_queue: queue.Queue[_WorkerFailure],
    ) -> None:
        try:
            with ACLModelRunner(self.artifact, device_id=self.device_id, use_async_stream=True) as runner:
                input_adapter = RuntimeInputAdapter(runner.input_spec)
                ready_queue.put(_WorkerReady(worker_id=worker_id, input_spec=runner.input_spec, output_specs=runner.output_specs))

                while not start_event.is_set():
                    if stop_event.is_set():
                        return
                    time.sleep(QUEUE_POLL_TIMEOUT_SECONDS)

                if warmup_steps > 0:
                    warmup_cycle = itertools.cycle(warmup_samples)
                    for _ in range(warmup_steps):
                        if stop_event.is_set():
                            return
                        preloaded = next(warmup_cycle)
                        sample = input_adapter.adapt(preloaded.input_fp16)
                        runner.infer(sample)
                warmed_queue.put(_WorkerWarmed(worker_id=worker_id))

                while not stop_event.is_set():
                    try:
                        task = input_queue.get(timeout=QUEUE_POLL_TIMEOUT_SECONDS)
                    except queue.Empty:
                        continue
                    sample = input_adapter.adapt(task.preloaded.input_fp16)
                    outputs, pure_infer_ms = runner.infer_with_timing(sample)
                    result = OrderedInferenceResult(
                        seq_id=task.seq_id,
                        sample_index=task.sample_index,
                        preloaded=task.preloaded,
                        outputs=outputs,
                        pure_infer_ms=pure_infer_ms,
                        dispatch_started_ns=task.dispatch_started_ns,
                        worker_id=worker_id,
                    )
                    while not stop_event.is_set():
                        try:
                            output_queue.put(result, timeout=QUEUE_POLL_TIMEOUT_SECONDS)
                        except queue.Full:
                            continue
                        break
        except Exception as exc:
            if not stop_event.is_set():
                error_queue.put(_WorkerFailure(worker_id=worker_id, phase="worker", error=exc))
                stop_event.set()

    def _dispatcher_main(
        self,
        preloaded_samples: list[PreloadedSample],
        repeat: int,
        workers: list[_WorkerState],
        stop_event: threading.Event,
        error_queue: queue.Queue[_WorkerFailure],
    ) -> None:
        try:
            seq_id = 0
            for _ in range(repeat):
                for sample_index, preloaded in enumerate(preloaded_samples):
                    if stop_event.is_set():
                        return
                    task = _InferenceTask(
                        seq_id=seq_id,
                        sample_index=sample_index,
                        preloaded=preloaded,
                        dispatch_started_ns=time.perf_counter_ns(),
                    )
                    input_queue = workers[seq_id % self._active_instances].input_queue
                    while not stop_event.is_set():
                        try:
                            input_queue.put(task, timeout=QUEUE_POLL_TIMEOUT_SECONDS)
                        except queue.Full:
                            continue
                        break
                    seq_id += 1
        except Exception as exc:
            if not stop_event.is_set():
                error_queue.put(_WorkerFailure(worker_id=-1, phase="dispatcher", error=exc))
                stop_event.set()

    @staticmethod
    def _poll_failure(error_queue: queue.Queue[_WorkerFailure]) -> _WorkerFailure | None:
        try:
            return error_queue.get_nowait()
        except queue.Empty:
            return None


__all__ = ["ACLConcurrentOrderedExecutor", "ACLModelRunner", "ACLRuntimeError", "OrderedInferenceResult"]
