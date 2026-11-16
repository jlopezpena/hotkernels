# argmin_f64 attempts

## 1. Scalar baseline

- **Idea:** simplest correct version: one element per step, `fcmp` for NaN and for strictly smaller.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point only; same shape as naive C.
