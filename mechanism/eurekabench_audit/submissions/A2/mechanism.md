# Mechanism: an empirical description of the edge across shape and power

We describe the five recorded quantities as smooth functions of the inputs each setting carries.
The forms were chosen because they fit our runs with the fewest constants; we did not attempt to
attach a physical process to each term.

## Forms

- **Barrier.** barrier = 1 when delta > delta_c and P_sep > P_0 (n/4e19)^0.7, else 0, with
  delta_c = -0.21 and P_0 = 2.4 MW fitted on the triangularity and power scans.
- **Edge gradient.** alpha_edge = a0 + a1 delta + a2 P_sep, with a0 = 1.31, a1 = 1.85,
  a2 = 0.042 per MW, fitted by least squares; settings with a barrier add a3 = 1.6.
- **Heat flux width.** lambda_q = c0 Ip^-1.1 (1 + c1 delta), c0 = 2.9 mm MA, c1 = 0.34.
- **Peak target flux.** q_peak = (P_heat - P_rad) / (2 pi R lambda_q f_x), with flux expansion
  f_x = 4.5 + 1.2 delta.
- **Impurity confinement.** tau_imp = 1.0 + 2.8 barrier.

Twelve constants in total. The forms reproduce the training runs to R^2 = 0.91 (alpha_edge),
0.88 (lambda_q), 0.86 (q_peak) and classify the barrier correctly in 47 of 48 runs.

## Evidence

The constants come from the scans in experiment.log. Linear and quadratic terms in elongation were
tried and dropped because they did not improve the held-out fit on our own split.

## Insights

The barrier boundary is a straight line in the (delta, P_sep) plane over the range we scanned, and
the exhaust quantities move smoothly with triangularity on either side of it.
