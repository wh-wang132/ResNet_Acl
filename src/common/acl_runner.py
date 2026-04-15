from __future__ import annotations

import time
from typing import Any

import acl
import numpy as np

from .artifact_scanner import ArtifactRecord, TensorSpec

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0
ITEMSIZE_TO_TENSOR_META: dict[int, tuple[np.dtype, int]] = {
    int(np.dtype(np.float16).itemsize): (np.dtype(np.float16), 10),
    int(np.dtype(np.float32).itemsize): (np.dtype(np.float32), 1),
}


class ACLRuntimeError(RuntimeError):
    """ACL 推理运行时初始化或执行失败。"""


class ACLModelRunner:
    def __init__(self, artifact: ArtifactRecord, device_id: int = 0):
        self.artifact = artifact
        self.device_id = int(device_id)
        self._context: int | None = None
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
        self._output_host_ptrs: list[int] = []
        self._model_input_spec: TensorSpec | None = None
        self._model_output_specs: tuple[TensorSpec, ...] = ()
        self._opened = False

    def __enter__(self) -> "ACLModelRunner":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._opened:
            return
        self._check_ret(acl.init(), "acl.init")
        self._check_ret(acl.rt.set_device(self.device_id), f"acl.rt.set_device({self.device_id})")
        self._context = self._unwrap_with_ret(acl.rt.create_context(self.device_id), "acl.rt.create_context")
        self._check_ret(acl.rt.set_context(self._context), "acl.rt.set_context")
        self._model_id = self._unwrap_with_ret(
            acl.mdl.load_from_file(str(self.artifact.model_path)),
            f"acl.mdl.load_from_file({self.artifact.model_path})",
        )
        self._model_desc = acl.mdl.create_desc()
        self._check_ret(acl.mdl.get_desc(self._model_desc, self._model_id), "acl.mdl.get_desc")
        self._load_model_specs()
        self._prepare_io()
        self._opened = True

    def close(self) -> None:
        if self._input_dataset is not None:
            self._destroy_dataset(self._input_dataset, self._input_buffers)
            self._input_dataset = None
        if self._output_dataset is not None:
            self._destroy_dataset(self._output_dataset, self._output_buffers)
            self._output_dataset = None
        for ptr in self._input_device_ptrs:
            self._check_ret(acl.rt.free(ptr), f"acl.rt.free(input_ptr={ptr})", raise_on_error=False)
        self._input_device_ptrs.clear()
        for ptr in self._output_device_ptrs:
            self._check_ret(acl.rt.free(ptr), f"acl.rt.free(output_ptr={ptr})", raise_on_error=False)
        self._output_device_ptrs.clear()
        self._input_sizes.clear()
        self._output_sizes.clear()
        for ptr in self._output_host_ptrs:
            self._check_ret(acl.rt.free_host(ptr), f"acl.rt.free_host(output_host_ptr={ptr})", raise_on_error=False)
        self._output_host_ptrs.clear()
        if self._model_desc is not None:
            self._check_ret(acl.mdl.destroy_desc(self._model_desc), "acl.mdl.destroy_desc", raise_on_error=False)
            self._model_desc = None
        if self._model_id is not None:
            self._check_ret(acl.mdl.unload(self._model_id), "acl.mdl.unload", raise_on_error=False)
            self._model_id = None
        if self._context is not None:
            self._check_ret(acl.rt.destroy_context(self._context), "acl.rt.destroy_context", raise_on_error=False)
            self._context = None
        self._check_ret(acl.rt.reset_device(self.device_id), f"acl.rt.reset_device({self.device_id})", raise_on_error=False)
        self._check_ret(acl.finalize(), "acl.finalize", raise_on_error=False)
        self._model_input_spec = None
        self._model_output_specs = ()
        self._opened = False

    def infer(self, input_array: np.ndarray) -> tuple[np.ndarray, ...]:
        outputs, _ = self.infer_with_timing(input_array)
        return outputs

    def infer_with_timing(self, input_array: np.ndarray) -> tuple[tuple[np.ndarray, ...], float]:
        self._ensure_opened()
        self._check_ret(acl.rt.set_context(self._context), "acl.rt.set_context")
        self._validate_input_array(input_array, self.input_spec)
        host_input = input_array.tobytes()
        self._check_ret(
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
        self._check_ret(
            acl.mdl.execute(self._model_id, self._input_dataset, self._output_dataset),
            "acl.mdl.execute",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        outputs: list[np.ndarray] = []
        for index, spec in enumerate(self.output_specs):
            self._check_ret(
                acl.rt.memcpy(
                    self._output_host_ptrs[index],
                    self._output_sizes[index],
                    self._output_device_ptrs[index],
                    self._output_sizes[index],
                    ACL_MEMCPY_DEVICE_TO_HOST,
                ),
                f"acl.rt.memcpy(device_to_host, output_index={index})",
            )
            output_bytes = acl.util.ptr_to_bytes(self._output_host_ptrs[index], self._output_sizes[index])
            output = np.frombuffer(output_bytes, dtype=spec.dtype).reshape(spec.shape).copy()
            outputs.append(output)
        return tuple(outputs), elapsed_ms

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
        input_ptr = self._unwrap_with_ret(
            acl.rt.malloc(input_size, ACL_MEM_MALLOC_HUGE_FIRST),
            f"acl.rt.malloc(input_size={input_size})",
        )
        input_buffer = acl.create_data_buffer(input_ptr, input_size)
        self._add_dataset_buffer(self._input_dataset, input_buffer, "input")
        self._input_device_ptrs.append(input_ptr)
        self._input_buffers.append(input_buffer)
        self._input_sizes.append(input_size)

        for index, spec in enumerate(self.output_specs):
            output_size = int(acl.mdl.get_output_size_by_index(self._model_desc, index))
            expected_output_size = self._expected_nbytes(spec)
            if output_size != expected_output_size:
                raise ACLRuntimeError(
                    f"OM 输出字节数与摘要不一致: output_index={index}, model={output_size}, summary={expected_output_size}"
                )
            output_ptr = self._unwrap_with_ret(
                acl.rt.malloc(output_size, ACL_MEM_MALLOC_HUGE_FIRST),
                f"acl.rt.malloc(output_size={output_size})",
            )
            output_host_ptr = self._unwrap_with_ret(
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
            self._check_ret(acl.destroy_data_buffer(buffer), "acl.destroy_data_buffer", raise_on_error=False)
        buffers.clear()
        self._check_ret(acl.mdl.destroy_dataset(dataset), "acl.mdl.destroy_dataset", raise_on_error=False)

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
    def _unwrap_with_ret(result: Any, action: str) -> Any:
        if not isinstance(result, tuple) or len(result) != 2:
            raise ACLRuntimeError(f"{action} 返回值格式异常: {result!r}")
        value, ret = result
        if ret != 0:
            raise ACLRuntimeError(f"{action} 失败，ret={ret}")
        return value

    @staticmethod
    def _check_ret(result: int, action: str, *, raise_on_error: bool = True) -> bool:
        if result == 0:
            return True
        if raise_on_error:
            raise ACLRuntimeError(f"{action} 失败，ret={result}")
        return False

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


__all__ = ["ACLModelRunner", "ACLRuntimeError"]
