# Code copied for `delivery apps`

Programs in this directory are other people's, copied unchanged. The wrappers in `delivery/apps/` supply their input and bound their run; they do not change a NumPy call.

| Directory | Is | From | Licence |
|---|---|---|---|
| `min_char_rnn/` | `min-char-rnn.py`, a character-level recurrent network by Andrej Karpathy | <https://gist.github.com/karpathy/d4dee566867f8291f086> | BSD, as the file's header says |
| `pyhpc/` | The NumPy implementations and input generators of three benchmarks of pyhpc-benchmarks, routines of the Veros ocean model: `equation_of_state`, `isoneutral_mixing`, `turbulent_kinetic_energy`. Each is its `__init__.py` and `*_numpy.py` from `benchmarks/<name>/` | <https://github.com/dionhaefner/pyhpc-benchmarks> at `d438ef8b76ecb51a019b105a6e06150e5a35c177` | Unlicense, in `pyhpc/LICENSE` |
