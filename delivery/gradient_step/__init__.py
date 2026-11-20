"""The gradient-step study: where the time of `w -= lr * g` goes in a training loop.

`uv run --group apps delivery gradient-step` runs it. It answers issue 191,
whether fusing the multiply and the subtraction of a gradient step is worth
having and whether hotkernels can reach it, and its findings are written up in
docs/gradient-step-findings.md. It measures the real applications of
`delivery apps`, `mlp_train` and `min_char_rnn`, with loops and an allocator
of its own built from the C files beside this module. macOS on ARM64 only:
the loops are NEON and the effect it measures belongs to that machine.
"""
