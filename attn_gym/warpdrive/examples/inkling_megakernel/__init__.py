"""Inkling 2 Mini (Thinking Machines' looped MoE) decode as warpdrive megakernel launches.
Modules:

- ``ref``: the PyTorch oracle (HF's eager ``Inkling2LoopForCausalLM`` numerics) and checkpoint
  loading;
- ``fast``: a CUDA-graph capturable decode step on swappable kernels;
- ``attn_kernel``, ``mlp_kernel``: a visit's attention and MoE MLP halves as one warpdrive
  launch each, with message gates in core passes after the first; ``msg_kernel``: a pass's
  messages; ``head_kernel``: the final block residual, norm and LM head; ``common``: their
  shared device code;
- ``megakernel``: the host side: the whole decode step as launches (``InklingDecoder``) and
  hybrid kernels that plug plain visits into ``fast``;
- ``bench``: CUDA-graph timings of a visit's halves and of the decode step;
- ``chat``: the model in the warpdrive chat (``attn_gym.warpdrive.examples.chat``).
"""
