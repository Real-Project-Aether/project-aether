# Mechanism: a linear filter from the sample to each motor neuron

We describe the membrane potential of each motor neuron as the sample passed through a fitted
linear filter plus a slow drift term. The form was chosen because it predicts the recorded activity
with few constants; we did not identify which cells or connections implement it.

## Form

For motor neuron class k (head, A, B, D):

  V_k(t) = V0_k + g_k * (h * s)(t - tau_k) + d_k * s_slow(t)

where s is the sample, h(t) = exp(-t/T) / T is a first-order low-pass kernel with a shared time
constant T, tau_k a class delay, g_k a class gain, s_slow the sample averaged over 10 s and d_k its
weight. Removing a kind of connection multiplies the gain by a fitted factor r_gap or r_chem.

Fitted constants (12): T = 0.41 s; tau = 0.18, 0.33, 0.29, 0.35 s; g = 3.1, 1.2, 1.4, 0.9 mV/pA;
d = 0.6 mV/pA (shared); r_gap = 0.93; r_chem = 0.12.

## Evidence

Fitted on 64 stimulation protocols (experiment.log). Held-out R^2 on our own split: 0.81 head,
0.74 A, 0.77 B, 0.69 D. Second-order kernels were tried and gave no improvement.

## Insights

A single shared low-pass time constant with class-specific delays and gains is enough to predict
the motor pattern; the gap-junction factor is close to one and the chemical factor close to zero.
