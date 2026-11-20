# Paper notes: FindingDory (Yadav, Ali, Gupta, Gal, Kira; ICLR 2026, arXiv:2506.15635)

Sources read on 2026-11-20: the arXiv PDF (main text plus Appendices B, C.1 and D), the official repo
`findingdory-benchmark/findingdory-train` (shallow clone, files `findingdory/*.py`), and the HF dataset
card for `yali30/findingdory-subsampled-96`. Section and figure numbers below refer to the arXiv PDF.

## 1. Research motivation
- Long-video QA benchmarks can often be solved from a few key frames, so they reward keyframe
  selection rather than long-term memory (Sec. 2.1). Embodied benchmarks either need exploration,
  which mixes memory up with exploration, or use one-hot or non-photorealistic tasks (Table 1).
- FindingDory separates memory from exploration with two phases (Sec. 3.1):
  1. **Experience collection.** A scripted oracle robot in Habitat/HSSD does 2–11 pick-and-place
     interactions. Episodes are 400–3500 frames long.
  2. **Interaction.** The agent receives the recorded history (images, pose, actions) and a templated
     instruction, and must navigate to the right place, e.g. "the receptacle you picked an object
     from right after interacting with the cup".

## 2. Main contributions
1. A benchmark of 60 task templates in 11 low-level categories. It has 79,213 train and 5,876 val task
   instances, over 1,478 train and 100 val episodes and 107/30 scenes.
2. Baselines built as a hierarchy: a high-level VLM picks goal frames, then a low-level ImageNav
   (OVRLv2) policy or a mapping policy navigates to them.
3. Analysis. Frozen VLMs score 13–27% high-level success rate (HL-SR). An SFT Qwen2.5-VL-3B scores
   about 50%. Multi-goal tasks are near zero for frozen VLMs. Frozen VLMs **do not improve and often
   get worse with more frames** (Fig. 4a), while the SFT model improves with longer videos.
4. A sim-to-real signal: co-training on FindingDory adds about 1.7–1.9 points on VSI-Bench (Table 3).

## 3. How observation history is represented
- The full history is an RGB video. Each frame has a burned-in overlay with the **original frame number
  and the time of day** (we confirmed this in the decoded 96-frame videos, e.g. "Frame: 302 / Time of
  Day: 07:12"). Pose and actions are logged but not used by the VLM baselines.
- The VLM baselines take the video directly. The Textual Memory Agent turns chunks of frames into JSON
  captions (room, action, object, receptacle, other objects), and an LLM then reasons over the text
  (App. D.4).

## 4. Existing frame-selection and memory baselines (what is already in the paper)
| Approach | How frames get chosen | Notes |
|---|---|---|
| Video LM agents (Gemini-2.0-Flash, GPT-4o, Qwen2.5-VL-7B, Gemma-3-12B, GLM-4.1V) | **Uniform subsampling** to a model-specific budget (Gemini at 1 fps; Qwen up to 768 frames; GPT-4o 96 frames) | No query-conditioned selection |
| Frame-budget sweep (Fig. 4a) | **Uniform** at 16 / 24 / 48 / 96 / 192 / 384 / 768 frames | The VLM then picks the goal frame from these |
| Textual Memory Agent | Every frame captioned in chunks, so nothing is dropped and the history is compressed into text | Worst baseline |
| SFT Qwen2.5-VL-3B | Uniform subsampling to 96 frames; ground truth remapped to the subsampled indices (App. D.5) | This is where the 96-frame dataset comes from |

**What the paper does not do, as far as we found:** no query-conditioned retrieval baseline (CLIP or
SigLIP top-k, embedding RAG), no use of temporal neighbourhoods around retrieved frames, and no
retrieval-only metric that scores the selected subset separately from the VLM's final answer. It cites
retrieval-style memory systems (ReMEmbR, Embodied-RAG, MobilityVLA; App. C.1) but does not evaluate
them. The authors point to "memory mechanisms" that extract task-relevant information from long
videos as future work (Sec. 5).

## 5. Ground truth and evaluation metrics
- **Ground-truth frames** are frames whose recorded pose satisfies the task's PDDL goal conditions:
  distance to goal, angle to goal, at least 0.1% semantic-mask coverage, and the room check
  (App. B.3–B.4). They are **viewpoints the robot could navigate from**, not "frames where the
  object appears". Each goal has a list of acceptable frames, and multi-goal tasks have one list per
  goal (App. D.5). In the 96-frame dataset these lists were remapped to subsampled positions 0–95.
- **Simulator metrics** (not reproducible without Habitat): HL-SR, LL-SR, HL-SPL, LL-SPL, DTG-SR and
  SC-SR. They are checked by teleporting the robot to the pose of the chosen frame.
- **Offline metrics in the repo** (`evaluate_llm_outputs.py`):
  - *Exact accuracy*: the predicted answer string equals the ground-truth string.
  - *Relaxed score*: pred and gt must have the same number of sublists. Per goal,
    precision = |pred ∩ gt| / |pred|, and the score is the product over goals.
  - **Quirk found in the code:** when one sublist is empty and the other is not, `precision = 0.0`
    is computed but **never appended** to `precision_all_goals`, so that goal is silently ignored.
    The `np.prod([])` of an all-empty match returns 1.0. We will not reuse this function as-is; our
    metrics count every goal explicitly.
- The paper's SFT evaluation at test time takes **one** predicted frame per goal and counts it correct
  if it is in that goal's list. This is close to our planned "any-hit" metric.

## 6. Temporal vs. spatial categories (from dataset labels)
- **Single-Goal Temporal**: Interaction Order (16 templates, e.g. "immediately after…", "2
  interactions before…"), Time-Based ("at 7:42"), Duration Tracking ("longest time to rearrange").
- **Single-Goal Spatial**: Object Recall, Interaction, Conditional Interaction, Object Attributes,
  Spatial Relationship ("farthest from your current location"), Room Visitation.
- **Multi-Goal**: Unordered Revisitation, Ordered Revisitation.

## 7. Limitations relevant to us
- In the paper, frame selection and reasoning are entangled: HL-SR measures the whole VLM.
- The 96-frame uniform subsampling already discards context. The authors note that a solution frame
  can survive subsampling while the context needed to recognise it does not (App. B.4).
- Spatial Relationship tasks ("farthest from current location") depend on the agent's current pose,
  which is not in the 96-frame dataset. Neither text nor images can identify the target.
- About 3.85% of tasks are "originally unsolvable" from frames (distractor or never-visited targets).
  These are the `[[-1]]` answers.

## 8. How our experiment differs
- **Question:** under a fixed budget k of frames chosen from the 96, does query-conditioned selection
  (CLIP top-k), and top-k with temporal neighbourhoods (TNR), cover the ground-truth goal frames
  better than uniform or random selection?
- The paper's frame-budget axis (Fig. 4a) uses only **uniform** subsampling and measures the
  **downstream VLM**. We measure **retrieval quality directly** against the ground-truth frame lists,
  with no VLM, and compare query-conditioned selectors at matched budgets. Our uniform baseline is the
  paper's selection strategy, re-applied inside the 96 frames.
- We are not claiming novelty for "retrieve frames with CLIP" (standard video-RAG practice), nor for
  temporal-context expansion in general (common in video moment retrieval). At most the contribution
  is a **controlled, budget-matched test of temporal expansion on an embodied-memory benchmark,
  broken down by task category**.

## 9. Assumptions that need validation
1. ~~Answer indices refer to positions 0–95 in the 96-frame video.~~ **Verified**: decoded length is
   96 and time-of-day overlays line up with Time-Based questions.
2. CLIP similarity between the question text and frames carries useful signal. **Doubtful for many
   templates**: the questions are about interaction logic ("second object", "farthest") rather than
   appearance, and many ground-truth frames are approach viewpoints (dark walls, gripper close-ups).
   This needs measuring, not assuming.
3. Ground-truth lists are contiguous runs (mean 1.5–4 runs per goal for most categories; 9 for
   "Interaction"). TNR neighbours of a correct anchor will often also be correct, which **inflates
   frame precision for TNR without any extra information**. Any-hit and coverage are the fairer
   primary metrics, and precision has to be read against the label structure.
4. Random selection is a strong any-hit baseline. The chance of at least one hit at k=8 is 38–92%,
   depending on category (Sec. 4 of dataset_notes.md). All results must be reported against this
   floor.
