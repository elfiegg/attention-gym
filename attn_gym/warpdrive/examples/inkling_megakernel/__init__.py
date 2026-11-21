"""Inkling 2 Mini (Thinking Machines' looped MoE) decode, toward a megakernel. Modules:

- ``ref``: the PyTorch oracle (HF's eager ``Inkling2LoopForCausalLM`` numerics) and checkpoint
  loading;
- ``fast``: a CUDA-graph capturable decode step on swappable kernels;
- ``attn_kernel``, ``mlp_kernel``, ``megakernel``: a visit's attention and MoE MLP halves as
  one warpdrive launch each, and the hybrid kernels that plug them into ``fast``;
- ``bench``: CUDA-graph timings of a visit's halves and of the decode step;
- ``chat``: the model in the warpdrive chat (``attn_gym.warpdrive.examples.chat``).
"""
