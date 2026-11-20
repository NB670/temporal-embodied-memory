# Phase 3 notes: visual memory and semantic similarity

## Configuration (cache key `ViT-B-32_laion2b_s34b_b79k_063d9559`)
| Setting | Value | Why |
|---|---|---|
| Encoder | OpenCLIP `ViT-B-32`, `laion2b_s34b_b79k`, 512-d | Small, fast, CPU-capable; a standard retrieval baseline |
| Overlay | Top-left box rows 0–90, cols 0–210 blacked out | Removes the rendered "Frame N / Time HH:MM" text (a non-scene shortcut) |
| Aspect | Pad 640×480 portrait frames to square, then CLIP resize/normalise | Center crop would drop about 12% at the top and bottom |
| Text | Strip "The robot's goal is: " | Carries no information; max question length is 48 tokens (no truncation) |
| Similarity | s_t = ⟨e_q, e_t⟩, both L2-normalised | Cosine similarity |
| Ties | Broken by lower frame index | Deterministic |

Hardware: Apple M3 Pro, 18 GB, MPS. Python 3.11.13, torch 2.15.0, open_clip 4.2.2. Encoding 30
episodes (2,880 frames) took 15.7 s, and 778 unique questions took 1.8 s. Cached reruns take 0.03 s.
Cache size is 8 MB. Selection latency for semantic top-k is about 0.02 ms per query.

## Diagnostic: how much does similarity tell us about ground truth?
We compute the AUROC of s_t for GT-union frames against all other frames, per question, average within
each episode, then across the 30 episodes. As a **control**, we score the same ground truth using a
different question from the same episode ("mismatched").

| Category | AUROC | Mismatched | Gap |
|---|---|---|---|
| Object Recall | 0.735 | 0.583 | **+0.152** |
| Object Attributes | 0.715 | 0.596 | **+0.118** |
| Conditional Interaction | 0.677 | 0.601 | +0.076 |
| Unordered Revisitation | 0.661 | 0.610 | +0.051 |
| Interaction Order | 0.636 | 0.614 | +0.022 |
| Time-Based | 0.622 | 0.600 | +0.021 |
| Interaction | 0.638 | 0.637 | +0.001 |
| Duration Tracking | 0.542 | 0.546 | −0.004 |
| Room Visitation | 0.500 | 0.525 | −0.024 |
| Spatial Relationship | 0.575 | 0.605 | −0.030 |
| Ordered Revisitation | 0.561 | 0.646 | −0.086 |
| **Overall** | **0.630** | **0.602** | +0.028 |

(Episode SDs are 0.06–0.15. These are 30 episodes, so the numbers are exploratory.)

**Interpretation**
1. Most of the above-chance AUROC is **not question-specific**. Any question from the episode ranks
   GT frames above chance (0.60), because GT runs sit on interaction segments (gripper plus object
   close-ups), which CLIP scores highly for almost any "navigate to / revisit object" text.
2. Question-specific signal exists mainly where the question names a **visual entity**: Object
   Recall, Object Attributes and Conditional Interaction.
3. For **temporal** categories the question adds almost nothing (+0.02). Semantic anchors there are
   effectively "some interaction frame". This bears directly on TNR: expanding around such anchors
   covers interaction segments, which may raise hit rates through the generic prior rather than
   through temporal reasoning. Phase 6 will need a baseline that isolates this.
4. For Ordered Revisitation the matched score is *below* the mismatched one. The ordinal wording
   ("first, second") seems to pull similarity toward the wrong frames.

## Qualitative observation (ep_2, task_1 "Navigate to a can")
Frame 52 clearly shows the can but is **not** GT. The can was later moved, and GT is its final
location (63–74). Object-recall labels therefore encode the *post-rearrangement* state, which is a
temporal dependency that semantic similarity cannot resolve.
