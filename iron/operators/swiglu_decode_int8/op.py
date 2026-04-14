# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

from iron.common import (
    CompositeOperator,
    AIERuntimeArgSpec,
)
from iron.operators.swiglu_base import chain_swiglu_artifacts
from iron.operators.gemv_int8.op import GEMVInt8
from iron.operators.silu.op import SiLU
from iron.operators.elementwise_mul.op import ElementwiseMul


class SwiGLUDecodeInt8Callable:
    """Callable for the W8A16 INT8 SwiGLU decode composite.

    Diverges from the bf16 ``_SwiGLUCallable`` in two ways:
      - Weights are pre-packed uint8 numpy buffers (INT8 weights + bf16 scales
        packed per tile, see ``gemv_int8/reference.py::quantize_and_pack``),
        uploaded as-is with no transpose.
      - GEMV_int8 argument order is ``(packed_weights, input, output)`` — same
        as bf16 GEMV but operating on the packed buffer.

    Activation, intermediates, and output remain bfloat16 (W8A16).
    """

    def __init__(self, op):
        def create_callable(xclbin_path, kernel_name, insts_artifact):
            return NPUKernel(
                xclbin_path=xclbin_path,
                kernel_name=kernel_name,
                insts_path=insts_artifact.filename,
            )

        combined = op.combined_xclbin.filename
        self.gemv_1_callable = create_callable(
            combined,
            op.gemv_int8_1_xclbin.kernel_name,
            op.gemv_int8_1_insts,
        )
        self.silu_callable = create_callable(
            combined,
            op.silu_xclbin.kernel_name,
            op.silu_insts,
        )
        self.eltwise_mul_callable = create_callable(
            combined,
            op.eltwise_mul_xclbin.kernel_name,
            op.eltwise_mul_insts,
        )
        self.gemv_2_callable = create_callable(
            combined,
            op.gemv_int8_2_xclbin.kernel_name,
            op.gemv_int8_2_insts,
        )

        # Upload packed INT8 weight buffers. These are numpy uint8 byte buffers
        # containing [INT8 weights][bf16 scales] per tile (see
        # ``gemv_int8/reference.py::quantize_and_pack``). No transpose — the
        # packer already lays them out in the DDR layout the design expects.
        # XRTTensor's array-like constructor path copies data into a host-only
        # XRT BO of matching size; dtype=np.uint8 keeps bytes exact.
        self.weights_1 = XRTTensor(
            np.asarray(op.weights_1, dtype=np.uint8), dtype=np.uint8
        )
        self.weights_2 = XRTTensor(
            np.asarray(op.weights_2, dtype=np.uint8), dtype=np.uint8
        )
        self.weights_3 = XRTTensor(
            np.asarray(op.weights_3, dtype=np.uint8), dtype=np.uint8
        )

        # Intermediate buffers remain bf16 in a W8A16 pipeline.
        intermediate_shape = (op.hidden_dim_padded,)
        self.left = XRTTensor(intermediate_shape, dtype=bfloat16)
        self.right = XRTTensor(intermediate_shape, dtype=bfloat16)
        self.left_swished = XRTTensor(intermediate_shape, dtype=bfloat16)
        self.intermediate = XRTTensor(intermediate_shape, dtype=bfloat16)

    def __call__(self, input_buf, output_buf):
        # Sync input in case caller wrote to it via torch_view() between calls.
        # Weights and internal buffers stay on device; the XRT runtime syncs
        # all args to device automatically before each kernel invocation.
        input_buf.to("npu")

        # 1. gemv_int8(weights_1 = packed gate, input) -> left
        self.gemv_1_callable(self.weights_1, input_buf, self.left)

        # 2. gemv_int8(weights_2 = packed up, input) -> right
        # Reuses gemv_1 kernel: gate and up projections share the same config
        # (both produce hidden_dim outputs from embedding_dim activation).
        self.gemv_1_callable(self.weights_2, input_buf, self.right)

        # 3. SiLU(left) -> left_swished  (bf16, unchanged from bf16 SwiGLU)
        self.silu_callable(self.left, self.left_swished)

        # 4. EltwiseMul(left_swished, right) -> intermediate  (bf16)
        self.eltwise_mul_callable(self.left_swished, self.right, self.intermediate)

        # 5. gemv_int8(weights_3 = packed down, intermediate) -> output
        self.gemv_2_callable(self.weights_3, self.intermediate, output_buf)


class SwiGLUDecodeInt8(CompositeOperator):
    """W8A16 (INT8 weights + bf16 activations) SwiGLU composite for decode.

    Wires two INT8 dequant-GEMV stages around a bf16 SiLU + EltwiseMul:
        output = down_i8 @ (SiLU(gate_i8 @ x) * (up_i8 @ x))

    Weights (``weights_1``/``weights_2``/``weights_3``) must be set to packed
    uint8 numpy buffers produced by
    ``iron.operators.gemv_int8.reference.quantize_and_pack`` before calling
    ``compile()``. weights_1 / weights_2 are the gate / up projections with
    shape ``(hidden_dim, embedding_dim)``; weights_3 is the down projection
    with shape ``(embedding_dim, hidden_dim)``. The packer is parameterised by
    the *clamped* ``tile_size_input`` from the GEMVInt8 instance — read it
    back after instantiation.

    Naming: class name is distinct from bf16 ``SwiGLUDecode`` and the inner
    sub-op prefixes are ``gemv_int8_1``/``gemv_int8_2``, so xclbin cache keys
    do not collide with any bf16 SwiGLU / GEMV / GEMM variants. SiLU and
    ElementwiseMul produce identical binaries to their bf16-chain counterparts
    by construction (both operate on bf16 intermediates), so cache sharing
    there is intentional and correct.
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        group_size=32,
        context=None,
    ):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        # Packed uint8 weight buffers to be set by user before .compile().
        # Use quantize_and_pack(M=target_M, K=target_K, group_size=group_size,
        # m_input=<gemv_int8>.tile_size_input, cols=num_aie_columns).
        self.weights_1 = None  # packed gate:  shape (hidden_dim, embedding_dim)
        self.weights_2 = None  # packed up:    shape (hidden_dim, embedding_dim)
        self.weights_3 = None  # packed down:  shape (embedding_dim, hidden_dim)

        super().__init__(context=context)

        # Eagerly declare sub-ops so callers can inspect gemv_int8_1 /
        # gemv_int8_2 to read back the *clamped* tile_size_input before
        # packing weights.  set_up_artifacts() is guaranteed side-effect-free
        # (no compilation); compile() guards against re-entry via
        # `if not self.artifacts`.
        self.set_up_artifacts()

    def set_up_artifacts(self):
        # Use the full column count available on the target device: 4 on NPU1
        # (Phoenix, aie2) and 8 on NPU2 (Strix, aie2p).
        n_cols = aie_utils.get_current_device().cols

        # Stage 1: gate / up projections. Both share the same GEMVInt8 config
        # (same M=hidden_dim, K=embedding_dim), so a single xclbin is compiled
        # and reused for both invocations at runtime.
        gemv_int8_1 = GEMVInt8(
            M=self.hidden_dim,
            K=self.embedding_dim,
            num_aie_columns=n_cols,
            tile_size_input=1,
            tile_size_output=self.hidden_dim // n_cols,
            group_size=self.group_size,
        )
        self.gemv_int8_1 = gemv_int8_1

        silu = SiLU(
            size=self.hidden_dim,
            num_aie_columns=n_cols,
            tile_size=self.hidden_dim // (n_cols * 2),
        )
        self.silu = silu
        self.hidden_dim_padded = silu.size

        eltwise_mul = ElementwiseMul(
            size=self.hidden_dim,
            num_aie_columns=n_cols,
            tile_size=self.hidden_dim // n_cols,
        )
        self.eltwise_mul = eltwise_mul
        assert self.hidden_dim <= eltwise_mul.size <= self.hidden_dim_padded

        # Stage 2: down projection.
        gemv_int8_2 = GEMVInt8(
            M=self.embedding_dim,
            K=self.hidden_dim,
            num_aie_columns=n_cols,
            tile_size_input=1,
            tile_size_output=self.embedding_dim // n_cols,
            group_size=self.group_size,
        )
        self.gemv_int8_2 = gemv_int8_2

        chain_swiglu_artifacts(
            self,
            [
                ("gemv_int8_1", "0x901", gemv_int8_1),
                ("silu", "0x902", silu),
                ("eltwise_mul", "0x903", eltwise_mul),
                ("gemv_int8_2", "0x904", gemv_int8_2),
            ],
        )

    def get_arg_spec(self):
        # Only input activation and output are exposed to the caller. The
        # three packed weight buffers are uploaded once inside the callable
        # and remain on device between invocations.
        return [
            AIERuntimeArgSpec("in", (self.embedding_dim,)),
            AIERuntimeArgSpec("out", (self.embedding_dim,)),
        ]

    def get_callable(self):
        return SwiGLUDecodeInt8Callable(self)
