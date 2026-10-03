# Mechanism: how the reversed shape keeps the edge out of the barrier regime

## The chain

1. **Shape to edge geometry.** Reversing the triangularity to -0.5 moves the outboard edge of the
   flux surfaces into the region where the field-line curvature is unfavourable over a larger
   fraction of each poloidal turn, and it changes the local magnetic shear along the field line on
   the low-field side.

2. **Geometry to ballooning stability.** With the larger unfavourable-curvature fraction and the
   altered local shear, the ideal ballooning modes at the edge lose access to the second stability
   regime. The edge can only sit on the first-stability side, so the normalised pressure gradient
   alpha at 0.95 is capped at the first ballooning limit.

3. **Stability to the barrier.** A transport barrier needs the edge gradient to rise above the
   first ballooning limit into the second-stable branch. Because that branch is closed, the
   barrier cannot form, and adding heating power does not change this: the cap is set by the
   geometry, not by the power. This is why no heating power brings the barrier back.

4. **No barrier, no bursts.** Without a pedestal there is nothing to drive peeling-ballooning
   bursts, so the edge stays free of edge-localised modes. Putting more of the plasma in the
   unfavourable-curvature region is exactly what closes second stability, so it is also what
   keeps the edge free of bursts.

5. **Edge to exhaust.** With the edge held at the ballooning limit, transport near the separatrix
   stays at the barrier-free level. The heat channel to the divertor narrows, the wetted area of
   the reversed divertor geometry is larger, impurities are not held, so the impurity confinement
   time stays close to the energy confinement time, and the regime survives strong core radiation
   because the cap does not depend on the power crossing the boundary.

6. **Critical triangularity.** The barrier stops forming where the second-stability window closes.
   That point moves with elongation and aspect ratio because both change the curvature fraction.

## Constants

The executable form uses a critical triangularity, a first-stability cap on alpha, and power-law
exponents for the exhaust quantities, fitted on our runs.

## Evidence

Our scans of triangularity, heating power and radiated fraction were consistent with every step of
this chain (see experiment.log). We considered a rising L-H power threshold as an alternative and
rejected it because the barrier did not return at higher power.

## Insights

The shape alone organises the edge: it fixes the ballooning limit, the limit fixes the gradient,
and the gradient fixes the exhaust.
