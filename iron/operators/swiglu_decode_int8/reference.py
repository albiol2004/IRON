# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Golden reference generator for the W8A16 SwiGLU decode composite.

Generates three quantized + packed weight buffers (gate, up, down) via
``iron.operators.gemv_int8.reference.quantize_and_pack`` and computes the
reference SwiGLU output against the *dequantized* weights — i.e. against what
the fused dequant-GEMV kernel actually sees after sign-extending the INT8
values and multiplying by the per-group bf16 scales. This keeps the reference
consistent with the kernel's numeric path and avoids double-counting the
quantization error as a correctness violation.
"""

import torch

from iron.operators.gemv_int8.reference import quantize_and_pack


def generate_golden_reference(
    embedding_dim=1024,
    hidden_dim=3584,
    group_size=32,
    m_input_gate=1,
    m_input_down=1,
    cols=4,
    seed=42,
):
    """Generate packed INT8 weights + bf16 golden output for SwiGLU decode (W8A16).

    The SwiGLU FFN computes: ``down @ (SiLU(gate @ x) * (up @ x))``

    Two distinct tile layouts are needed because ``gate``/``up`` share one
    GEMVInt8 config (M=hidden_dim, K=embedding_dim) while ``down`` uses a
    different config (M=embedding_dim, K=hidden_dim). Both tile layouts are
    driven by the GEMVInt8 instance's *clamped* ``tile_size_input`` (returned
    from ``__post_init__``) — callers must read that back from the operator
    and pass it in here rather than using the raw default.

    Args:
        embedding_dim: Model embedding dim (K for gate/up, M for down).
        hidden_dim:    FFN hidden dim  (M for gate/up, K for down).
        group_size:    Quantization group size (default 32, matches GEMVInt8).
        m_input_gate:  ``tile_size_input`` used by the gate/up GEMVInt8.
        m_input_down:  ``tile_size_input`` used by the down GEMVInt8.
        cols:          Number of AIE columns the composite spans.
        seed:          RNG seed.

    Returns:
        dict containing:
          ``x``            -- bf16 (embedding_dim,) input activation
          ``packed_gate``  -- numpy uint8 packed weights for the gate proj
          ``packed_up``    -- numpy uint8 packed weights for the up proj
          ``packed_down``  -- numpy uint8 packed weights for the down proj
          ``W_gate``       -- bf16 (hidden_dim, embedding_dim)  dequantized gate
          ``W_up``         -- bf16 (hidden_dim, embedding_dim)  dequantized up
          ``W_down``       -- bf16 (embedding_dim, hidden_dim)  dequantized down
          ``left``         -- bf16 (hidden_dim,) = gate @ x
          ``left_swished`` -- bf16 SiLU(left)
          ``right``        -- bf16 (hidden_dim,) = up @ x
          ``intermediate`` -- bf16 (hidden_dim,) = left_swished * right
          ``output``       -- bf16 (embedding_dim,) = down @ intermediate
    """
    torch.manual_seed(seed)

    # Random input activation. Same amplitude as gemv_int8 reference so the
    # quantized levels get fully exercised.
    val_range = 4
    x = torch.randn(embedding_dim, dtype=torch.bfloat16) * val_range

    # Pack each of the three projections. ``quantize_and_pack`` uses a fresh
    # torch.randn internally (seeded by the global manual_seed set above), so
    # gate / up / down receive *distinct* random weights in sequence.
    packed_gate, W_gate = quantize_and_pack(
        M=hidden_dim,
        K=embedding_dim,
        group_size=group_size,
        m_input=m_input_gate,
        cols=cols,
    )
    packed_up, W_up = quantize_and_pack(
        M=hidden_dim,
        K=embedding_dim,
        group_size=group_size,
        m_input=m_input_gate,
        cols=cols,
    )
    packed_down, W_down = quantize_and_pack(
        M=embedding_dim,
        K=hidden_dim,
        group_size=group_size,
        m_input=m_input_down,
        cols=cols,
    )

    # Compute golden output against the dequantized weights so the reference
    # matches the kernel's numeric path. All matmuls are promoted to fp32
    # to mirror the NPU's accfloat accumulation before the final bf16 round.
    x_f32 = x.to(torch.float32)
    left = (W_gate.to(torch.float32) @ x_f32).to(torch.bfloat16)
    left_swished = torch.nn.functional.silu(left)
    right = (W_up.to(torch.float32) @ x_f32).to(torch.bfloat16)
    intermediate = (left_swished * right).to(torch.bfloat16)
    output = (W_down.to(torch.float32) @ intermediate.to(torch.float32)).to(
        torch.bfloat16
    )

    return {
        "x": x,
        "packed_gate": packed_gate,
        "packed_up": packed_up,
        "packed_down": packed_down,
        "W_gate": W_gate,
        "W_up": W_up,
        "W_down": W_down,
        "left": left,
        "left_swished": left_swished,
        "right": right,
        "intermediate": intermediate,
        "output": output,
    }
