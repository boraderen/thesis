# Analysis guide

Practical heuristics on top of the thesis and the kairo docs. The parameters, their values and
defaults are in the docs; what references and signal window sizes do is in the approach chapter.

## 1. Starting points

Rules of thumb to start with, not results. Validate them with the rows per window and the
clustering scores.

- **Volume.** From `compute_log_stats`: span S (days), cases and events per day, median case
  duration.
- **Feature window Δf** (resource, inter-case): the shortest calendar unit (`1h`, `6h`, `12h`,
  `1D`, `3D`, `7D`) whose median window still holds about 20–30 events (resource) or 5 active
  cases (inter-case), with few empty windows and a few hundred rows in total.
- **Signal window Δs:** about 20–60 signal windows over the span (Δs ≈ S/50, rounded to a calendar
  unit). Intra-case: at least about 10–20 cases per window. Resource and inter-case: Δs = Δf
  mirrors the log trajectory, Δs ≈ 7–30 × Δf measures shares. Real logs: multiples of seven days,
  so every window holds the same weekdays.
- **Scaling:** z-score for resource and inter-case (mixed units), min-max for intra-case.
- **PCA:** fit all components, cut at the elbow of the reconstruction error. Inter-case with few
  columns: try without PCA first.
- **Clustering:** SOM with a small grid (e.g., 3 × 3) and a large one for exploration (6 × 6 to
  10 × 10); k-means with k = 2 … 8 by silhouette; DBSCAN with `min_samples` = 2 × clustering
  dimensions and `eps` at the knee of the k-distance curve.
- **Drift signal:** KL with `previous` first, then confirm with `recent` (lookback 3–7) and a
  bounded divergence (JS or TV).

## 2. Tuning

- Change one parameter at a time and compare scores on the same rows (helper: `sweep_kmeans`,
  `sweep_som`, `sweep_dbscan`, `k_distance_summary`).
- **Large SOM → k:** groups of occupied cells between dark ridges of the u-matrix are candidate
  states; count them or cluster the occupied prototypes (`suggest_k_from_som`), then use that k.
- Bigger SOM grids always lower the quantization error; watch the occupancy entropy and the used
  cells. A grid that gives almost every distinct row its own cell is too big.
- A too large k or grid splits stable periods, and the state changes look like drifts. Prefer the
  smallest k that separates the behaviors you see.
- DBSCAN: a small `eps` gives many tiny clusters and much noise, a large one merges clusters and
  hides the drift. Check the noise share around suspected drifts.
- Let each run set the next one: a spike → zoom into its date range with shorter signal windows;
  a state that only occurs after a date → its profile tells which features matter; a quiet
  perspective → check whether its features can see the suspected drift type at all.
- A real drift persists across nearby window lengths, references and divergences. A spike in one
  configuration only is an artifact until shown otherwise.

## 3. Reading a run

1. **Is the configuration meaningful?** Warning signs: one state holds almost all rows, many empty
   SOM cells, silhouette below about 0.25, DBSCAN mostly noise or one giant cluster, a PCA cut
   dominated by a few rare z-scored columns, thin signal windows. Fix these before reading the
   signal.
2. **Which states change?** Frequencies per date range (`get_*_state_frequencies(states, start,
   end)`, `plot_som_heatmap`), trajectories before and after a date.
3. **What do they stand for?** Unscaled feature means per state (`state_profiles`), PCA loadings,
   for intra-case the prefixes of the events in a state.
4. **What changed?** The largest share changes (`abstract_state_distributions`) and feature shifts
   before and after a date (`feature_shift`). List the top windows with rows, coverage and new
   cases (`signal_window_report`, `top_windows`).
5. **Mode:** one spike → sudden; raised scores over several windows with old and new states side
   by side → gradual; repeated spikes with returning states → recurring (or a calendar pattern).

## 4. Pitfalls

| Pitfall | Check |
|---|---|
| Partial first or last window (calendar windows start at multiples from 1970-01-01; at the end no new arrivals) | Rows, coverage and new cases of the window; compare with the full windows next to it |
| KL and a state missing by chance in a thin reference window | Confirm with JS or TV and with `recent`; check the rows of the reference |
| Empty or quiet feature windows (nights, weekends, holidays) become z-score outliers, own states or noise and look like recurring drifts | Longer feature windows, multiples of seven days, or compare without them |
| DBSCAN noise share rises after a drift because rows spread | Read the noise share separately; look at the noise rows' features |
| PCA cut by variance drops a low-variance feature that carries the drift | Check loadings; try a larger cut or no PCA |
| z-score inflates rare columns, min-max is squeezed by one outlier | Compare both on the same features |
| Attribute drifts are invisible without `case_attributes` | Map every case attribute the log has |
| `case_progress` uses the length of the whole case | Mention it if the log has running cases |

## 5. Playbooks

Patterns, not scripts; adapt them to the task.

- **Find drift points and perspectives.** Initial runs in all required perspectives → date ranges with high
  scores (check for truncation) → zoom in and tune where the states are not meaningful → explain
  each candidate by states and features → if several perspectives react at the same date, decide
  the drift perspective with the feature figures of the approach chapter (direct versus side
  effect) → mode from the shape → confirm with other windows, references and divergences.
- **Check one drift point in one perspective.** Choose signal windows that put the point well
  inside a window → compare state frequencies, distributions and the signal before and after →
  vary windows, references, divergences and feature groups that can see the expected drift type →
  look at the other perspectives too → answer yes, no or unsure with the evidence.
- **Date ranges, states, events, cases, attributes.** Filter the log or pass `start_date` and
  `end_date` (cases cut by a filter change their prefixes); look up the states of specific events
  in the intra-case states frame (same index as the log); compare single-case trajectories; map
  attributes as `case_attributes`.
- **Compare configurations.** Fix the features and windows, vary one thing, compare scores on the
  same rows and the drift signals. Prefer meaningful states and a robust signal over the highest
  score.
- **Cross-perspective question.** Find the drift in X, then run Y with feature groups that can see
  the side effect and with ones that cannot. A side effect starts at the same time or later, never
  earlier.

## 6. Custom features (only with `custom allowed`)

How to pass an own frame is in the kairo docs (*Flexible by design*). In addition:
- name the columns `group:name`, fill missing values (usually 0) and keep the index unique;
- window rows are equally spaced and floored like kairo's (`dt.floor`), one row per window from
  the first to the last, empty ones included;
- with the helper: `kp.run_pipeline(config, log=log, features=custom)`.

## 7. Thresholds

For every reported configuration, exactly one for each required perspective (`SKILL.md`, rule
11), suggest one drift score above which a signal window counts as a drift, for a later
evaluation:
- Split the signal windows into the ones you report as drifts and the others; leave out the first
  and last windows and windows flagged as thin or partial.
- Take a value between the highest score of the others and the lowest score of the drift windows,
  rounded to two significant digits, e.g., `KL ≥ 2.0`.
- Keep the range of the divergence in mind: KL from 0 to about 20.72 (kairo's `1e-9` smoothing),
  where a state missing on one side alone gives a high score; JS from 0 to ln 2 ≈ 0.693; TV and
  Hellinger from 0 to 1.
- With the `baseline` reference, the signal shows levels instead of spikes; the threshold then
  separates the two levels.
- If no value separates them, give the value that keeps all drift windows and say in the results
  how many other windows it also flags.
- If the perspective shows no drift, still report its configuration with a threshold just above
  the highest score of its scored windows, rounded up to two significant digits, so that it flags
  no window; say in the results that the perspective shows no drift.

## 8. Drift signal files

Save a plot and a CSV for every row of the configuration table, before you delete your run
folder. Both get the name `<log name>_<perspective>` (`SKILL.md`, *Results files*) and go to your
new result folder, `analyses/<benchmark>/<log name>/<k>/`: the plot to `plots/` with the threshold
as a dashed line, the CSV to `drift_signals/`. The helper claims the result folder and creates the
subfolders:

```python
folder = kp.new_result_folder(log_path)   # once per analysis, e.g., analyses/padella/log_07/3
files = kp.save_drift_signal(result["signals"][i], folder, "resource", threshold)
```

The CSV holds one row per signal window: `signal_window` (its start), `drift_score` (empty for the first window), `rows`, `coverage`
and `flags` of the signal window report.

Without the helper:

```python
folder = "analyses/padella/log_07/3"   # your claimed result folder; mkdir folder/plots folder/drift_signals first
name = "log_07_resource"
fig = kairo.plot_drift_signal(signal, divergence, reference)   # signal from compute_drift_signal
fig.add_hline(y=threshold, line_dash="dash")
fig.write_image(f"{folder}/plots/{name}.png", width=900, height=300)
report.reset_index().to_csv(f"{folder}/drift_signals/{name}.csv", index=False)   # signal_window_report
```

## 9. Checklist before the entry

- [ ] Every reported configuration is meaningful (section 3, step 1).
- [ ] Every drift point appears with at least two signal window lengths or references, or the entry
      says it does not.
- [ ] High scores in the first and last windows are checked and explained.
- [ ] KL spikes are confirmed with a bounded divergence or the reference window is checked.
- [ ] The states that change are named and profiled.
- [ ] The perspective of each drift is argued, and the reactions of the others are explained.
- [ ] Calendar effects, thin or empty windows and outliers are ruled out or mentioned.
- [ ] Exactly one configuration for each required perspective is reported (two for a Padella log:
      intra-case and resource; three for every other log: intra-case, resource and inter-case), the
      others that confirm it are named in the *Analysis* bullets.
- [ ] Every number comes from a run of this analysis, and every configuration has a threshold and a
      saved plot and CSV.
