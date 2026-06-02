# Attack Plan — Match *structure*, not *coordinates*, in FOA-Attack

## The one-line idea

FOA-Attack makes an adversarial image's local features land on the **same
absolute positions** as the target image's features (in the surrogate model's
feature space). The problem: those absolute positions are specific to *that*
model, so they don't carry over to the closed-source model we actually want to
fool. Instead, match the **pattern of relationships between regions** — which
region looks like which — because that pattern is shared across different vision
models. Same goal (make the model describe the image as the target), but aligned
on the part of the representation that actually transfers.

---

## Background: what FOA-Attack already does

We want a perturbed image that a closed-source multimodal model (GPT-4o, Claude,
Gemini) describes as some chosen *target* thing instead of what it really is. We
can't see those models' gradients, so we optimize against a few open CLIP models
(the "surrogates") and hope it carries over ("transfers").

FOA-Attack's recipe, in plain terms:

1. A vision model cuts the image into a grid of small patches and produces one
   feature vector per patch ("local features" / patch tokens).
2. There are hundreds of these and many are redundant, so it **groups them into
   a few representative summary vectors** — group the patch features into `n`
   groups and take each group's average (this grouping is called *K-means*,
   `n` = number of groups, e.g. 3 or 5).
3. It then finds the **cheapest way to match** the adversarial image's `n`
   summary vectors onto the target image's `n` summary vectors, where "cost"
   means "how dissimilar two vectors are." This cheapest-matching step is called
   *optimal transport*, solved with the *Sinkhorn* algorithm; the matching it
   returns is the *transport plan*.
4. The loss is the total matching cost. Pushing it down drags the adversarial
   summary vectors onto the target's summary vectors.

This works on absolute feature values. That's the weak point.

---

## Why this is the weak point

Two different vision models are like two people who both recognise a cat but use
**different internal vocabularies** to describe its parts. If you force the
adversarial image to produce the exact feature *values* the target produces in
person A's vocabulary, person B (different vocabulary) won't agree — the values
mean nothing to them.

What the two people *do* agree on is the **pattern of relationships**: the
eye-region relates to the ear-region in the same way for both of them, even
though the raw numbers differ. Studies that compare what's shared between two
neural networks (this line of work is called *representational similarity* /
*CKA*) find exactly this: absolute feature values differ a lot between models,
but the pattern of "which thing is similar to which" is largely shared.

And here is the subtle part worth heading off — *doesn't FOA's clustering
already take care of structure?* No. Clustering cleans up the **inputs** (it
hands the matcher a few coherent region-groups instead of hundreds of noisy
individual patches), but it does not change what the **loss rewards**. The
matching step still scores each adv group by one question only — "how close is
its absolute feature value to its matched target group's?" — judging every group
**on its own address**. The geometry *between* groups never enters the loss, no
matter how you batch the patches. So clustering is FOA's only nod to structure,
and it's a nod to the inputs, not to the objective.

So FOA matches the part that *doesn't* transfer (absolute coordinates) and
ignores the part that *does* (relationships). That's the gap.

---

## How matching relationships actually works (the mechanism)

Once you see what Gromov-Wasserstein (the "match-by-internal-structure" tool,
GW for short) *aims the perturbation at*, the rest follows. The name is jargon;
the idea is simple.

**The metaphor.** Optimal transport (FOA) gives directions like **GPS
coordinates** — "go to latitude 40.7, longitude −74.0." That only works if the
other person uses your exact map grid. Two different vision models are two cities
that each laid down their own private grid: the same coordinates point to
different places. GW gives directions by **relationships** — "find the spot where
the school sits next to the park, the park is far from the factory, and the
factory faces the school." Those directions need *no shared grid*; they work in
any city that has that layout. GW describes the target by its **internal floor
plan**, not by addresses only the surrogate understands.

**Concretely, with three regions.** Say the target is a cat, and the surrogate
turns it into three region-summaries: an ear-region, an eye-region, a
nose-region. What GW cares about is the cat's **internal distance map** — how
similar each region is to each other region:

```
            ear   eye   nose
   ear       -    0.3   0.7
   eye      0.3    -    0.2      (small number = the two regions look alike)
   nose     0.7   0.2    -
```

The *shape* of this is the cat's relational signature: one pair very alike
(eye–nose, 0.2), one moderate (0.3), one far apart (0.7).

- **What FOA aims at:** "make adv-region-1's features *equal* the ear-region's,
  adv-region-2 *equal* the eye-region's, adv-region-3 *equal* the nose-region's"
  — three absolute addresses in the surrogate's grid.
- **What GW aims at:** "reshape the adversarial image until *its own* internal
  distance map looks like that triangle — one pair very close, one moderate, one
  far — and work out for yourself which adv region plays which role." GW never
  compares an adv feature to a target feature directly. It compares
  **adv-to-adv distances against target-to-target distances** — distances to
  distances.

**Why this survives the model swap (the whole point).** Optimizing the GW loss
sculpts the adversarial image so its *internal geometry* copies the cat's: two
regions become near-twins, one becomes the odd one out, with exactly those gaps.
Now hand that image to the closed-source model. It encodes everything in **its
own** vocabulary, so the absolute feature values come out completely different —
which is exactly why FOA's address-matching leaks away here. But the **internal
geometry is preserved**, because geometry is the part two models agree on (the
representational-similarity / CKA finding). The victim measures the image's
regions against each other, finds the cat's relational signature staring back —
in its own coordinate system — and describes it as a cat. You supplied the
structure; the victim supplied the vocabulary.

In one line: **FOA shouts a street address into a city that uses a different
grid; GW describes a floor plan that any grid can host.**

(One honest caveat, expanded in the Risks section: a floor plan alone is
ambiguous — different images can share the same layout — so we keep a little of
FOA's address-matching to break the tie. That's the "fused" version.)

---

## The change: align relationships, not positions

Keep everything in FOA the same — same surrogates, same step rule (PGD /
MI-FGSM), same crops, same escalation budget — and **only swap the local loss**.
Two versions, cheap and principled.

### Notation (so the pseudo code is unambiguous)

- `adv_patches`, `target_patches`: per-patch feature vectors, shape
  `[num_patches, dim]`.
- `X = [n, dim]`: the `n` summary vectors of the adversarial image.
- `Y = [n, dim]`: the `n` summary vectors of the target image.
- `cosine(A, B)`: matrix of cosine similarities between rows of A and rows of B.
- Gradient flows through `X` (it depends on the perturbation). `Y` is fixed.
  The matching plan is solved **once per step and then treated as fixed** (no
  gradient through the solver) — same as FOA already does.

### What FOA does now (for reference)

```python
def foa_local_loss(adv_patches, target_patches, n):
    X = kmeans_centers(adv_patches,    n)   # [n, dim]  adv summary vectors
    Y = kmeans_centers(target_patches, n)   # [n, dim]  target summary vectors

    cost = 1 - cosine(X, Y)                 # [n, n]  how dissimilar a is from b
    plan = sinkhorn(cost)                   # [n, n]  cheapest soft matching

    return (cost * plan).sum()              # minimizing -> X lands on Y
```

### Version A — cheap relational term (low risk, easy to run first)

Idea: make the adversarial image's **internal similarity pattern** equal the
target's. We need to know which adv summary lines up with which target summary —
just reuse FOA's matching plan for that.

```python
def relational_loss(adv_patches, target_patches, n):
    X = kmeans_centers(adv_patches,    n)   # [n, dim]
    Y = kmeans_centers(target_patches, n)   # [n, dim]

    # internal pattern: how each summary relates to the others, within one image
    S_X = cosine(X, X)                      # [n, n]  adv's internal relationships
    S_Y = cosine(Y, Y)                      # [n, n]  target's internal relationships

    # line up adv summaries with target summaries using FOA's matching
    cost  = 1 - cosine(X, Y)
    plan  = sinkhorn(cost).detach()         # fixed correspondence, no gradient
    order = plan.argmax(dim=1)              # adv summary a  <->  target summary order[a]

    # reorder the target pattern into the adv ordering, then make them equal
    S_Y_aligned = S_Y[order][:, order]      # [n, n]
    return ((S_X - S_Y_aligned) ** 2).mean()
```

Only `S_X` carries gradient, so pushing this down reshapes the adversarial
image's internal relationships to mirror the target's.

### Version B — the principled one: match positions *and* relationships together

Instead of borrowing FOA's matching and then comparing relationships as a
separate step, solve for **one** matching that simultaneously (1) puts matched
summaries near each other and (2) preserves the relationship pattern. The tool
that does both at once is *Fused Gromov-Wasserstein* — "Gromov-Wasserstein" is
the matching-by-internal-structure part, "fused" means we also keep the ordinary
position-matching part. A weight `alpha` trades between them.

```python
def fused_structural_loss(adv_patches, target_patches, n, alpha):
    X = kmeans_centers(adv_patches,    n)   # [n, dim]
    Y = kmeans_centers(target_patches, n)   # [n, dim]

    # (1) position cost: dissimilarity between adv summary a and target summary b
    feat_cost = 1 - cosine(X, Y)            # [n, n]   <-- this is FOA's term

    # (2) internal relationship patterns
    S_X = cosine(X, X)                      # [n, n]
    S_Y = cosine(Y, Y)                      # [n, n]

    # find ONE matching that keeps positions close AND preserves relationships:
    #   if adv (a, a') map to target (b, b'),
    #   then S_X[a, a'] should equal S_Y[b, b'].
    # ready-made solver: POT library, ot.gromov.fused_gromov_wasserstein2
    plan = solve_fused_gw(feat_cost, S_X, S_Y, alpha).detach()   # [n, n], fixed

    position_term     = (feat_cost * plan).sum()
    relationship_term = structure_mismatch(S_X, S_Y, plan)       # see note below
    return (1 - alpha) * position_term + alpha * relationship_term

# relationship_term, written out:
#   sum over a, a', b, b' of  (S_X[a,a'] - S_Y[b,b'])^2 * plan[a,b] * plan[a',b']
```

Key property for a clean paper: **`alpha = 0` is exactly FOA-Attack.** So the
comparison is maximally fair — identical pipeline, one loss term generalized, and
`alpha` is the whole story.

### Where it plugs in

Replace the body of `EnsembleFeatureLoss_OT_foa_attack` (their local loss). Leave
the global cosine loss, the ensemble weighting, the crops, and the attack loop
untouched. Nothing else in FOA changes, which is what makes the comparison fair.

---

## Why the FOA authors probably didn't already do this

Not because it's obvious-and-rejected — because it's outside the frame they were
working in. Their innovation axis was **global vs local** ("CLS token ignores the
patches, so add the patches"). They never questioned whether matching *absolute
feature values* was itself the weak link. Optimal transport entered this line as
a nicer feature-matcher; matching-by-internal-structure comes from a different
toolbox (shape and graph correspondence) and a different motivation (transfer the
*shared* part of the representation). Looks one step away from outside; it isn't
one step away from where they were standing.

---

## The real risk to watch (this is the gate, not novelty)

The danger is **not** "someone would have thought of it." The danger is **it
might not actually beat FOA**, in which case `alpha` wants to be 0 and the new
term earns no weight. Specific things to expect:

- **`alpha` collapses to 0** → relationships don't help → no contribution. This
  is the make-or-break, and the pilot below tests it directly.
- **`alpha = 1` (pure structure) probably underperforms.** Matching only
  relationships pins the target down only up to a rotation/reflection of the
  feature space — for a *targeted* attack that can align to a mirror-image of the
  target that decodes to the wrong thing. The position term (`alpha < 1`) anchors
  it. So expect the best `alpha` somewhere in the middle, not at either end.
- **The full solver is heavier and can be finicky** to differentiate through. If
  it's unstable, fall back to Version A (the cheap relational term).
- **The grouping jitters** — K-means can group differently each step, adding
  noise. Fix the random seed or warm-start each step from the previous grouping.

---

## Plan, in order

1. **Lit check (the only thing that can kill novelty).** Search whether any
   vision-language transfer-attack paper already uses Gromov-Wasserstein /
   structural / relationship matching. If yes, re-scope. If no, proceed.
2. **Pilot (the efficacy gate).** ~20–100 images. Compare FOA (`alpha = 0`)
   against the fused version with a small sweep of `alpha`. Question: does the
   best `alpha` sit clearly above 0, and does closed-source attack success rate
   (and average description similarity) go up? Cheap to answer; settles the bet.
3. **If positive → full run.** Same benchmark as FOA, everything identical except
   the loss term. Headline ablation = the `alpha` sweep (0 = FOA, 1 = pure
   structure, middle = fused). Include Version A as a second, simpler baseline so
   the contribution isn't staked on one temperamental solver.
4. **Secondary axis (only after the core works).** Do the structural match at a
   few grouping sizes (`n` in, say, {3, 5, 8}) and combine — now "multi-scale" is
   *motivated* two ways rather than being an arbitrary average over a knob:
   relationships are hierarchical (object → parts → sub-parts), and a finer
   grouping hands GW a **bigger distance map** — a richer floor plan to copy — so
   a larger `n` finally has a reason behind it.

---

## The thesis sentence, for the paper

> Transferable targeted attacks should align the part of the representation that
> is shared across vision models — the pattern of relationships between local
> regions — rather than the absolute feature values, which are specific to the
> surrogate and don't survive transfer to a closed-source model.
