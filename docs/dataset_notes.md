# Dataset notes: `yali30/findingdory-subsampled-96`

Inspected on 2026-11-20 at revision `95afe1e8ef355e06f27000851fc4033266bdd480` (last modified
2025-07-07). License: Apache-2.0. Reproduce with `python -m src.data.inspect_dataset --n-episodes 30 --seed 0`.

## 1. Files
| File | Size | Contents |
|---|---|---|
| `data/train-00000-of-00001.parquet` | 2.0 MB | 79,218 questions, 1,478 episodes, 55 task ids |
| `data/validation-00000-of-00001.parquet` | 0.2 MB | 5,876 questions, 100 episodes, 60 task ids |
| `videos.zip` | 3.19 GB | `videos/{train,val}/ep_<n>.mp4` (1,478 + 100 files) |
| `episode_id_mappings.json` | small | `{split: {original_id: renamed_id}}` |

**`video` is a relative path string** (`videos/val/ep_1.mp4`), not media bytes. The videos live only
inside `videos.zip`. The official code downloads and extracts the whole zip (`findingdory/utils.py`).
We instead read the zip's central directory with HTTP Range requests (0.12 MB) and pull single members
of about 2 MB each (`RemoteZipVideoFetcher`). The 30-episode subset needed about 82 MB of transfer and
takes 66 MB on disk.

## 2. Schema (matches the card)
`ep_id, video, question, answer, task_id, high_level_category, low_level_category, num_interactions`.
Every field is a string except `num_interactions` (int64). `answer` is a **stringified nested list**,
`"[[5, 6, 7], [54, 55]]"`, so we parse it with `ast.literal_eval`.

## 3. Videos
- All 30 subset videos: **96 frames, 640×480 (H×W, portrait), 1.0 fps**, mp4, decoded with OpenCV.
- Decoded frame *i* is dataset index *i* (answers range from −1 to 95).
- Each frame has a burned-in overlay in the top-left: `Frame: <original full-video index>` and
  `Time of Day: HH:MM`. The overlay number is **not** the 0–95 index (ep_2: 0, 6, 12, … 612).
- Alignment check: ep_2 task_48, "object you interacted with at 7:42", has ground truth 63–74. The
  overlays on frames 64 and 74 read 07:38 and 07:54. ✓
- Caveat: the overlay text means CLIP can partly "read" times. That may help Time-Based questions
  in a way that has nothing to do with scene semantics.

## 4. Answer structure (validation split)
| answer_kind | Definition | Count | Where |
|---|---|---|---|
| `frames` | every goal has ≥1 valid index | 5,595 | all categories |
| `unseen` | `[[-1]]`: target was never observed | 226 | Room Visitation "room you did not visit" (100), Spatial Rel. "receptacle you did not interact with…farthest" (69), Conditional "X you did not interact with" (51), Interaction (6) |
| `empty` | `[[]]`: no valid frame survives | 55 | Conditional (37), Spatial Rel. (15), Interaction (3) |
| `partial` | multi-goal with a `[-1]` or `[]` goal | 0 in val (34 in train) | Ordered/Unordered Revisitation |

- Single-goal questions always have exactly one inner list. Multi-goal questions have 1–20 lists
  (mean ≈5). The order of the lists follows the instructed order for Ordered Revisitation tasks.
- **284 of 779 multi-goal questions have overlapping goal lists**: the same frame is valid for two
  goals, e.g. one receptacle used twice. A single selected frame can therefore cover two goals. Goal
  coverage must be computed per goal, not on the flattened set.
- `unseen` and `empty` cannot be scored as frame retrieval. They go to a separate analysis
  (out of scope for any-hit; possibly an "abstain" analysis later).

**Ground-truth geometry (answer_kind = frames, validation):**
| low-level category | n | goals | valid frames (of 96) | runs per goal | P(random k=8 hits ≥1) |
|---|---|---|---|---|---|
| Spatial Relationship | 396 | 1 | 6.0 | 1.5 | 0.38 |
| Object Attributes | 491 | 1 | 7.0 | 1.8 | 0.43 |
| Interaction Order | 1530 | 1 | 7.5 | 2.0 | 0.43 |
| Conditional Interaction | 596 | 1 | 8.0 | 2.3 | 0.45 |
| Time-Based | 200 | 1 | 8.1 | 2.0 | 0.45 |
| Object Recall | 200 | 1 | 10.8 | 3.1 | 0.55 |
| Duration Tracking | 299 | 1 | 18.1 | 2.8 | 0.57 |
| Interaction | 591 | 1 | 29.2 | 9.0 | 0.88 |
| Room Visitation | 392 | 1 | 31.5 | 4.0 | 0.84 |
| Unordered Revisitation | 600 | 5.2 | 26.4 (union) | 1.8 | 0.81 (any goal) |
| Ordered Revisitation | 300 | 6.0 | 29.6 (union) | 1.7 | 0.93 (any goal) |

Takeaways: labels are **short contiguous runs of about 6–13 frames**, typically one interaction
episode from approach to departure. Runs include approach frames that show only walls or floor (see
the figure). Random any-hit at k=8 is already 38–93%, so **any-hit saturates for broad categories**.
The informative comparisons are the narrow-label categories, small k (4), and multi-goal coverage.

## 5. Episodes and splits
- `ep_id` is **only unique within a split**: train and val both contain `ep_1`…`ep_100`, which are
  different videos. We key episodes by `(split, ep_id)`.
- Validation has 51–60 questions per episode (mean 58.8), all sharing one video, with no duplicate
  (episode, task) pairs. Questions in an episode are **not independent**, so bootstraps must
  resample episodes.
- `episode_id_mappings.json`: 53 *original* ids occur in both the train and val maps. We can't tell
  whether these are different scenes that happen to share numbering or a real overlap. Since we
  train nothing, this does not affect us. **Unverified.**
- Validation has exactly 10 episodes for each `num_interactions` value from 2 to 11. Train is skewed
  toward few interactions.
- Object Attributes (5 templates) exist only in validation.

## 6. Our subset and split policy
- **Split: official `validation` only.** It contains all 60 templates and is the paper's evaluation
  split. We train nothing. If we ever tune a hyperparameter (e.g. TNR's anchor fraction or α), we
  tune it on a few *train* episodes, so the validation results stay held out.
- **30 episodes**, 3 per `num_interactions` stratum, sampled with `seed=0`. Selection uses metadata
  only, before any model is run (`data/metadata/subset_episodes.json`).
- Subset: 1,758 questions, 1,668 of them with `frames` answers (Spatial 789, Temporal 609,
  Multi-goal 270), 69 `unseen` and 21 `empty`.
- **Update (Phases 6–7):** embedding all 100 validation episodes was cheap (82 s), so the final
  experiments use **all 100** (5,595 evaluable questions). The 30-episode subset became the *dev*
  subset that was inspected during development. The other 70 episodes give a held-out check
  (`results/tables/e1_paired_k8_heldout70.csv`). All 100 videos decode to exactly 96 frames with no
  invalid frames.

## 7. Mismatches with the original proposal
1. `video` holds paths, not bytes. Handled with the range-request downloader.
2. Labels are viewpoint-feasibility sets, not "object visible" sets, and they come as contiguous runs.
   Frame precision is biased in favour of temporal expansion (neighbours of a hit are usually hits).
3. Many templates contain no visual content to match, e.g. "the second object you interacted with" or
   "farthest from your current location". Semantic retrieval may work no better than uniform there.
   That is a legitimate finding, but we should expect it.
4. Spatial Relationship needs the agent's *current pose*, which is not available. We report it
   separately and flag it as not solvable from frames and text.
5. The overlay text (frame number and time) is visible to the image encoder.
