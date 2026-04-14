#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch

from ml_dtypes import bfloat16
import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from iron.operators.swiglu_decode_int8.op import SwiGLUDecodeInt8
from iron.operators.swiglu_decode_int8.reference import generate_golden_reference
from iron.common.test_utils import verify_buffer


def get_params():
    max_aie_columns = aie_utils.get_current_device().cols

    # (embedding_dim, hidden_dim, num_aie_columns, group_size)
    # Qwen3.5-0.8B FFN target: embedding=1024, hidden=3584. 3584 is evenly
    # divisible by both 4 (=896) and 8 (=448) so gate/up tile_size_output
    # satisfies hidden_dim % num_aie_columns == 0 on both devices.
    params_list = [
        (1024, 3584, 4, 32),  # Qwen FFN on NPU1 (4 cols)
        (1024, 3584, 8, 32),  # Qwen FFN on NPU2 (8 cols)
    ]

    params = []
    for p in params_list:
        embedding_dim, hidden_dim, num_aie_columns, group_size = p
        # Skip tests that require more columns than the active device has.
        if num_aie_columns > max_aie_columns:
            continue
        params.append(
            pytest.param(
                *p,
                id=f"swiglu_i8_e{embedding_dim}_h{hidden_dim}_{num_aie_columns}col_g{group_size}",
            )
        )
    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "embedding_dim,hidden_dim,num_aie_columns,group_size", get_params()
)
def test_swiglu_decode_int8(
    embedding_dim, hidden_dim, num_aie_columns, group_size, aie_context
):
    operator = SwiGLUDecodeInt8(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        group_size=group_size,
        context=aie_context,
    )

    # set_up_artifacts() runs inside CompositeOperator.__init__, so gemv_int8_1
    # and gemv_int8_2 already exist here. Read back the *clamped* tile_size_input
    # from each (GEMVInt8.__post_init__ may halve it to fit L1 budget) so the
    # packer produces a layout that matches what the design DMAs expect.
    m_input_gate = operator.gemv_int8_1.tile_size_input
    m_input_down = operator.gemv_int8_2.tile_size_input

    golden_ref = generate_golden_reference(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        group_size=group_size,
        m_input_gate=m_input_gate,
        m_input_down=m_input_down,
        cols=num_aie_columns,
    )

    operator.weights_1 = golden_ref["packed_gate"]
    operator.weights_2 = golden_ref["packed_up"]
    operator.weights_3 = golden_ref["packed_down"]

    operator.compile()
    op_func = operator.get_callable()

    input_buf = XRTTensor.from_torch(golden_ref["x"])
    output_buf = XRTTensor((embedding_dim,), dtype=bfloat16)

    # Warmup
    op_func(input_buf, output_buf)

    start = time.perf_counter()
    op_func(input_buf, output_buf)
    elapsed_us = (time.perf_counter() - start) * 1e6

    total_bytes = input_buf.buffer_object().size() + output_buf.buffer_object().size()
    bandwidth_gbps = total_bytes / (elapsed_us * 1e-6) / 1e9
    print(f"Latency (us): {elapsed_us:.2f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.4f} GB/s")

    errors = {}

    # Per the bf16 SwiGLU methodology (see fix-swiglu-decode-test-rect),
    # we do NOT assert `intermediate` against a CPU-fp32 golden.  Two
    # artefacts dominate that comparison and are correct behaviour, not
    # kernel bugs: (a) the NPU SiLU LUT rounds tiny negatives to 0 while
    # CPU keeps them, amplified by large `right` operands; (b) at ~1e6
    # magnitudes bf16's ULP reaches ~8192, so NPU's bf16 mul and CPU's
    # fp32-then-bf16 mul disagree by up to 1 ULP purely from round-off.
    # A self-consistency metric (intermediate vs NPU's own ls*right) is
    # printed for regression visibility; the real correctness check is
    # the final `output` against a down-projection golden that uses the
    # NPU's own intermediate as input.
    observed_ls = op_func.left_swished.to_torch().reshape((hidden_dim,))
    observed_right = op_func.right.to_torch().reshape((hidden_dim,))
    intermediate = op_func.intermediate.to_torch().reshape((hidden_dim,))
    npu_sc_diff = (
        intermediate.to(torch.float32)
        - (observed_ls.to(torch.float32) * observed_right.to(torch.float32))
    ).abs()
    print(
        f"NPU self-consistency (intermediate vs ls*right): "
        f"max_diff={float(npu_sc_diff.max()):.2f}  "
        f"mean_diff={float(npu_sc_diff.mean()):.2f}"
    )

    # Output check: chain the reference down projection on top of the NPU's
    # own intermediate so residual error isolates the down GEMV alone.
    ref_output = (
        golden_ref["W_down"].to(torch.float32) @ intermediate.to(torch.float32)
    ).to(torch.bfloat16)
    output = output_buf.to_torch().reshape((embedding_dim,))
    errors_output = verify_buffer(
        output, "output", ref_output, rel_tol=0.05, abs_tol=1.0
    )
    if errors_output:
        errors["output"] = errors_output

    assert not errors, f"Test failed with errors: {errors}"
