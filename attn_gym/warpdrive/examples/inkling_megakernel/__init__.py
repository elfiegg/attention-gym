"""Inkling 2 Mini (Thinking Machines' looped MoE) decode as warpdrive megakernel launches.
Modules:

- ``ref``: the PyTorch oracle (HF's eager ``Inkling2LoopForCausalLM`` numerics) and checkpoint
  loading;
- ``fast``: a CUDA-graph capturable decode step on swappable kernels;
- ``visit_kernel``: a visit (its attention half ``attn_half``, then its MoE MLP half
  ``mlp_half``) as one warpdrive launch, with message gates in core passes after the first;
  ``msg_kernel``: a pass's messages; ``head_kernel``: the final block residual, norm and LM
  head; ``common``: their shared device code;
- ``quant``: the checkpoint's NVFP4/MXFP8 experts kept quantized in the expert GEMV's layout,
  their exact bf16 dequantization and the oracle on such a model; ``quant_gemv``: the register
  decode the visit kernel's expert GEMVs use on them; ``formats``: the formats and tile
  geometry both share (no imports, so the loader needs no CuTeDSL);
- ``megakernel``: the host side: the whole decode step as launches (``InklingDecoder``);
- ``bench``: CUDA-graph timings of each launch kind and of the decode step;
- ``chat``: the model in the warpdrive chat (``attn_gym.warpdrive.examples.chat``).
"""
