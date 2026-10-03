# Mechanism: the reversed shape raises the power needed to form a barrier

## The chain

1. **Shape to the ion drift geometry.** Reversing the triangularity moves the X-point radially
   inward relative to the outboard midplane, which changes the direction in which the ion
   grad-B drift carries ions across the last closed flux surface relative to the X-point. In the
   reversed shape this drift points away from the favourable side.

2. **Drift geometry to the power threshold.** The L-H power threshold rises when the ion drift
   points away from the X-point. Our scans show the threshold rising steeply as delta falls,
   from 2.4 MW at delta = +0.5 to 9.8 MW at delta = -0.3 and above 20 MW at delta = -0.5.

3. **Threshold to the barrier.** At the 6 MW of the reference case the reversed shape is far
   below its threshold, so no barrier forms. With enough power the barrier would return; the
   reversed shape is therefore an L-mode by power deficit.

4. **No barrier, low edge gradient.** Without a barrier, edge transport is set by L-mode
   turbulence, which holds alpha_edge low. The gradient stops at the value the L-mode transport
   allows for the given power.

5. **Exhaust.** The L-mode edge carries a narrower heat channel; impurities are flushed at the
   L-mode rate, so their confinement time stays near the energy confinement time; core radiation
   lowers the power crossing the boundary and keeps the plasma further below threshold.

6. **Critical triangularity.** The critical triangularity is where the threshold crosses the
   available power. It shifts with elongation and aspect ratio because they change the X-point
   position.

## Constants

delta-dependence of the threshold (two constants), L-mode alpha scaling (two), exhaust exponents
(four).

## Evidence

Threshold scan per triangularity (experiment.log, run set 2) and the barrier returning at 24 MW for
delta = -0.5 (run set 3). Ballooning stability was checked and rejected as the cause because the
edge gradient in the reversed shape stays well below the ballooning limit.

## Insights

The reversed shape is a high-threshold configuration: its freedom from the barrier is bought by
keeping the power below a threshold that the shape itself raises.
