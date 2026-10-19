# Entry template for results.md

Write one entry per task to the `results.md` of your own new result folder
(`analyses/<benchmark>/<log name>/<k>/`, `SKILL.md`, *Results files*), in one command
(`cat > analyses/<benchmark>/<log name>/<k>/results.md <<'EOF' … EOF`). The file holds only this
entry. Never change the `results.md` of another result folder.

Style:
- Every bullet is one short sentence. Keep to the bullet limits below.
- Every field on its own line.
- The file starts with `# Analysis results: <log name>`, the entry heading follows and starts with the timestamp from the system clock (`date "+%Y-%m-%d %H:%M"`) at the
  time you write the entry.
- Dates as `YYYY-MM-DD`, signal windows by their start, state names as kairo prints them, drift
  scores with three decimals.
- No config ids or hashes and no clustering scores (silhouette, Calinski–Harabasz, DBCV,
  quantization error, occupancy entropy) anywhere in the entry.
- Configuration table: at most one configuration per perspective, one row each with exactly one
  drift signal and one threshold. If several configurations show the drifts of a perspective,
  take the one whose threshold separates the drift windows most clearly and name the others in the
  *Analysis* bullets. Plain text in the cells, each wrapped in `<small>…</small>` to keep the
  table small. For a Padella log, only intra-case and resource rows.
- Distance: the distance measure of the clustering (`euclidean`, `cosine`, `manhattan` or
  `chebyshev`); k-means always uses `euclidean`.
- Threshold: for each configuration, the drift score above which a signal window counts as a
  drift, chosen as in `skills/kairo-drift-analyst/references/analysis-guide.md` (*Thresholds*).
  Use the numeric form `<divergence> ≥ <number>` in the threshold cell; benchmarks read this
  table directly for the specified run rather than choosing among configurations.
- Drift signals: one plot and one CSV per row of the table, saved before you delete your run
  folder (same guide, *Drift signal files*) as `plots/<name>.png` and `drift_signals/<name>.csv`
  in your result folder, with `<name>` = `<log name>_<perspective>`. The plots are embedded with
  their path relative to `results.md` (`plots/<name>.png`) in the order of the table.

```markdown
# Analysis results: <log name>

## <YYYY-MM-DD HH:MM> · <log file name> · <short task title>

- **Agent:** <agent>, <model>
- **Task:** <the task in one sentence>
- **Log:** `<path>`; <cases> cases, <events> events, <start> to <end>
- **Features:** <built-in only | custom allowed>
- **Custom features:** <only if used: which and why, in one sentence>

### Configurations

| <small>Perspective</small> | <small>Features</small> | <small>Scaling, PCA</small> | <small>States</small> | <small>Distance</small> | <small>Signal</small> | <small>Threshold</small> |
|---|---|---|---|---|---|---|
| <small>intra-case</small> | <small>act_freqs, df_counts, current_act</small> | <small>min-max, d=8</small> | <small>SOM 3×3, lr 0.5</small> | <small>cosine</small> | <small>28D, JS, recent (lookback 3)</small> | <small>JS ≥ 0.15</small> |
| <small>resource</small> | <small>res_events, act_res_shares, Δf 1D</small> | <small>z-score, d=1</small> | <small>k-means, k=2</small> | <small>euclidean</small> | <small>7D, KL, previous</small> | <small>KL ≥ 2.0</small> |

### Drift signals

![intra-case drift signal, 28D JS recent](plots/<log name>_intra.png)

![resource drift signal, 7D KL previous](plots/<log name>_resource.png)

### Results

- <at most 4 bullets, e.g., "Sudden resource drift in the week of 2021-06-10 (KL 20.723): two
  activities moved to other resources.">

### Analysis

- <at most 3 bullets, e.g., "Weekly windows with the recent reference and JS kept the same week.">
  (other configurations that confirm a drift are named here, not in the table)

### Pros and cons

- <at most 2 bullets, e.g., "Con: daily intra-case windows were too thin for this log.">
```
