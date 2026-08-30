# L5, a new kind of move: not buildable now, and here is precisely why

We did not build an L5 mechanism. This records the reasoning, because "we ran out of time" and
"there is nothing to build against" are different claims and only the second one is true.

## 1. The evidence base is one event, and it is flagged uncertain

One event in 1,547: **Cohen 1966, forcing**, recorded as `L2|L5?` — the question mark is the
annotator's own doubt, and the note says it is "the corpus's strongest candidate for a new KIND
of transformation, though cited only for the independence result."

Across the 163 mathematics prizes — Fields, Abel, Wolf, Shaw, where inventing a *method* rather
than proving a *theorem* is routine — the depth spread is L0 46, L1 1, L2 59, L3 46, L4 18,
**L5 1**. Where we would most expect to find more, the scheme finds none. Either L5 is genuinely
almost absent from awarded work, or our annotator cannot recognise it. Both readings say the same
thing about building against it: there is nothing here to calibrate an operation on.

## 2. Every rung we did build has a measurable trigger. L5 has none.

| rung | fires when | measured by |
|---|---|---|
| L1 | a direction moves no prediction | null space of the prediction Jacobian |
| L3 | a structure-preserving map exists **and pays** | probe agreement, then the transferred claim is tested |
| L4 | a conserved quantity fails to balance | residual against the noise scale |
| **L5** | **the existing repertoire is insufficient** | **— nothing** |

The others are triggered by something the system can measure about the world. L5's trigger is a
statement about the system's own operations, and a repertoire cannot measure its own
insufficiency from the inside. That is not a gap in our engineering; it is the shape of the
problem.

## 3. The nearest attempt is already refuted, in this repository

Proposing new machinery from structural defects in a description — dangling structure, role
collisions, signature strain — is what `discovery/pressure.py` did, and it **lost on three
substrates**. The obvious L5 trigger, "we keep failing in a patterned way, so invent an
operation," is that idea one level up. Building it again without new evidence would be repeating
a refuted experiment with a bigger claim attached.

## 4. It needs the prerequisite the paper says is missing

Forcing is a method for *constructing models*. You can only invent a method for constructing
models if you already have a formal notion of what a model is. That is exactly the boundary in
the paper's Section VI: four of eight Standard Model concepts were bought by a theorem before any
experiment, and no discovery system carries a theory it can prove things about. L5 sits on the
far side of that line, not near it.

## What would change this

- **Re-annotation surfacing more L5 events.** If independent annotators find method-invention
  where we recorded L2 or L4, there is a population to work with. One uncertain example is not.
- **A formal substrate.** Give a system a theory it can reason about — a proof assistant, a type
  theory — and "invent a new kind of move" becomes tactic synthesis, which is a real and studied
  problem. That is a different project from this one, and an honest route to L5.
