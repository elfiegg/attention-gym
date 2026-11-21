"""Inkling 2 Mini (Thinking Machines' looped MoE) decode, toward a megakernel. Modules:

- ``ref``: the PyTorch oracle (HF's eager ``Inkling2LoopForCausalLM`` numerics) and checkpoint
  loading;
- ``fast``: a CUDA-graph capturable decode step on swappable kernels;
- ``mlp_kernel``, ``megakernel``: a visit's MoE MLP half as one warpdrive launch, and the
  hybrid kernels that plug it into ``fast``;
- ``chat``: the model in the warpdrive chat (``attn_gym.warpdrive.examples.chat``).
"""
