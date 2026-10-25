---
name: kairo-drift-analyst
description: Agentic concept-drift analysis of one event log with the kairo library (state-based process monitoring in the intra-case, resource and inter-case perspectives). Use when asked to find, verify or explain concept drifts, drift points, drift perspectives or state changes in an event log, or to tune kairo pipelines for that. Every analysis gets a new numbered result folder in the log's own folder under analyses/ (1, 2, 3, ...) with results.md, its drift signal plots and CSVs; the entry also names the agent and the model.
metadata:
  compatibility: Needs the thesis repository with kairo and its dependencies already installed. Run Python through `uv run python`; do not install packages during analysis.
---

# Kairo Drift Analyst

You are a data analyst specialized in process mining and concept drifts. You analyze one event
log with the Python library **kairo**, interpret the states, state frequencies, divergences and
drift signals of your pipeline runs, and report what you found in the `results.md` of a new
result folder of the log (see *Results files*). The thesis and the kairo docs explain the method and the
API; this skill only adds how to run an analysis.

## Inputs

The user, or an orchestrating agent, always gives these in the prompt:

| Input | Example |
|---|---|
| Event log | `data/thesis/eval/padella/log_07.xes` |
| Task | "find the drift points and their perspectives" |
| Agent and model | "Claude Code, Opus 5.5" |
| Features | `built-in only` (kairo's feature groups) or `custom allowed` (also own feature matrices) |

Work autonomously and do not ask the user. Write the agent and model into the entry exactly as
given.

## Rules

1. **Analyze only the given log.** Never swap in another log.
2. **Stay blind to the ground truth** (see *Forbidden files*).
3. **No LLM calls.** Never use `LLMConnector`, `create_messages`, `count_*_tokens`,
   `get_response_text` or `get_chat_history`. The `abstract_*` and `plot_*` functions are allowed
   and encouraged.
4. **Context first.** Finish the reading list (or the small-context path) before the first
   pipeline run.
5. **Features as the user says.** With `built-in only`, use kairo's feature groups (any selection
   and parameters). With `custom allowed`, you may also compute your own features from the
   `read_log` frame; the entry then says which, how and why.
6. **Double-check every claim.** Report nothing you have not verified. If unsure, leave it out or
   say so.
7. **Distrust the first and especially the last signal windows.** Check what is in them and
   describe it.
8. **Explain cross-perspective effects.** Say which perspective a drift belongs to and why other
   perspectives react.
9. **Traceable.** Every reported result comes from a run of this analysis, and the configuration
   table of the entry describes that run.
10. **One brief entry per task** in the `results.md` of your own new result folder; earlier result
    folders stay untouched. Report **exactly one configuration and one threshold for each
    required perspective** (rule 11), each with its drift signal: the one that shows the drifts
    of the perspective most clearly. Other configurations that confirm it go into the *Analysis*
    bullets only. Select it without ground truth or benchmark scores; later benchmarks evaluate
    this exact configuration and threshold from the specified result folder.
11. **Required perspectives.** For a Padella log (`data/thesis/eval/padella/`), exactly two:
    intra-case and resource, never inter-case. For every other log, exactly three: intra-case,
    resource and inter-case. So the configuration table, the plots and the CSVs have exactly two
    or exactly three entries. The required perspectives do not depend on the task or the
    findings: report each of them also if the task asks about one perspective only or if a
    perspective shows no drift. For a perspective without a drift, report its most meaningful
    configuration with a threshold that no scored signal window reaches (guide, *Thresholds*),
    and say so in the results.

The hard rules (`skills/kairo-drift-analyst/references/hard-rules.md`) always apply as well:
Python only through `uv`, no installs, logs only through kairo, one run folder in `analyses/`
that you delete at the end, one new result folder for the outcome, a progress file that survives
compactions. Read them first.

## Reading list

Paths from the repository root.

1. `thesis/chapters/prelim.tex`: formal definitions (event logs, windows, drift modes, scaling,
   PCA, clustering, scores, divergences).
2. `thesis/chapters/related_work.tex`: concept-drift detection and state-based process mining.
3. `thesis/chapters/approach.tex`: the framework. Its feature figures show which feature groups
   respond to which drift type, directly or through a side effect; the drift-signal section
   explains references, signal window sizes and the end of the log. The drift types themselves
   are in `thesis/chapters/rheon.tex` (section on drift types).
4. kairo docs: `kairo/index.html` (every function, feature group, window rule and how to pass own
   features, section *Flexible by design*). Where they are short, read the source in
   `kairo/kairo/`. `kairo.DEFAULT_SYSTEM_PROMPT` summarizes strengths and limits per perspective.
5. Useful examples: `notebooks/kairo_examples/`, `dashboard/`.

Two facts the docs leave implicit: kairo adds `1e-9` to every share, so KL is finite with a
maximum of `log(1e9) ≈ 20.72`; SOM state names `"(i, j)"` count from 0.

## Forbidden files

Never open, parse or list anything that reveals how a generated log was built or which drifts it
holds:
- metadata files such as `*_meta.md`;
- files next to logs that describe their changes, e.g., `README.md` or `change.png`;
- files that map renamed logs to their originals;
- the folders with the ground truth, `data/thesis/padella_with_ground_truth/` and
  `data/thesis/rheon_with_ground_truth/`;
- generator scripts such as `generate_*.py` and `data/generate_log.py`;
- benchmark results that compare detections with the injected drifts, i.e., everything in
  `benchmarks/`.

Do not try to identify the log by comparing it with other logs. Earlier entries of other logs
show the format only; never reuse their conclusions. If you are unsure whether a file
reveals the ground truth, do not open it.

## Results files

The folder of the log decides where its results go:

| Log folder | Analysis folder |
|---|---|
| `data/thesis/eval/padella/` | `analyses/padella/` |
| `data/thesis/eval/rheon_single_drift/` | `analyses/rheon_single/` |
| `data/thesis/eval/rheon_multi_drift/` | `analyses/rheon_multi/` |
| `data/thesis/eval/rheon_mixed_drift/` | `analyses/rheon_mixed/` |
| any other folder | `analyses/other/` |

Every log has its own folder in it, named like the log file without its extension, e.g.,
`analyses/padella/log_07/`. Every analysis of the log gets its own numbered result folder in it:
`1/` for the first, `2/` for the second, and so on, e.g., `analyses/padella/log_07/3/`. A result
folder holds the outcome of exactly one analysis:

| Path | Content |
|---|---|
| `results.md` | the entry of this analysis |
| `plots/` | the drift signal plots, one PNG per row of the configuration table: two for a Padella log, three for any other log |
| `drift_signals/` | the drift signals, one CSV per row of the configuration table: two for a Padella log, three for any other log |

Create your result folder only when you write the entry (*Workflow*, step 6), with the next number
after the highest one in the folder of the log. Claim it with a plain `mkdir` (no `-p`), which
fails if another agent took the number first; then try the next one. The helper does both:
`folder = kp.new_result_folder(log_path)`. In the shell:

```bash
base=analyses/padella/log_07; mkdir -p "$base"; k=$(( $(ls "$base" | grep -E '^[0-9]+$' | sort -n | tail -n 1) + 1 )); while ! mkdir "$base/$k" 2>/dev/null; do k=$((k + 1)); done; echo "$base/$k"
```

Never write into, change or delete another result folder. A plot and its CSV share the name
`<log name>_<perspective>`, with the perspective `intra`, `resource` or `inter`, e.g.,
`log_07_resource.png` and `log_07_resource.csv`. The guide (*Drift signal files*) says how to
write them.

## Workflow

0. **Intake.** Check the inputs. Create your run folder
   `analyses/<agent>-<YYYY-MM-DD-HHMM>-<log file name>/`, with your agent name in lowercase (e.g.,
   `codex`) and the time from `date "+%Y-%m-%d-%H%M"`. Other agents may work in parallel: check
   `ls analyses/` first, add `-2`, `-3`, … if the name is taken, and never touch other folders.
   Everything of this run goes there: your own `.py` scripts (run them with
   `uv run python analyses/<run folder>/<script>.py`), configs, helper outputs, figures and
   `progress.md`. Write down the task in one sentence and what "done" means in `progress.md`.
1. **Context.** Work through the reading list, or the small-context path.
2. **Reconnaissance.** Read the log with `kairo.read_log`, map every role column it has (resource,
   event duration, start timestamp), compute `kairo.compute_log_stats`, and check span, volume,
   resources, case attributes, gaps, and how the log starts and ends. A Padella log
   (`data/thesis/eval/padella/`) has no duration column: read it with
   `start_timestamp="start:timestamp"` and, before computing any features, add the event duration
   in minutes from its two timestamps:
   `log["event_duration"] = (log["time:timestamp"] - log["start_timestamp"]).dt.total_seconds() / 60`.
   This is log preparation, not a custom feature. With the helper, set
   `"event_duration_from_timestamps": true` in the `log` part of the config.
3. **Initial runs.** One configuration for each required perspective (rule 11), derived from the log
   statistics (`skills/kairo-drift-analyst/references/analysis-guide.md`, *Starting points*).
   Keep the run log in `progress.md`: the configuration, what changed, why, what you expected,
   what you saw.
4. **Iterate** until the task is answered: form a hypothesis, choose the next configuration to
   test it (feature groups and parameters, windows, scaling, PCA cut, clustering method and
   parameters, divergence, reference, lookback, date range), use the clustering scores and earlier
   runs to set parameters, and find which states and features change.
5. **Verify** with the checklist in the guide.
6. **Write the entry** with `skills/kairo-drift-analyst/references/report-template.md`: pick
   exactly one configuration for each required perspective (rule 11) and choose its threshold
   (guide, *Thresholds*), claim
   your result folder (*Results files*), save the drift signal plots and CSVs into it (guide,
   *Drift signal files*), then write the entry to its `results.md`.
7. **Clean up.** Delete your run folder (`rm -r analyses/<your run folder>`) and nothing else.

## Files

| File | Content |
|---|---|
| `skills/kairo-drift-analyst/references/hard-rules.md` | Rules that always apply, also after a compaction |
| `skills/kairo-drift-analyst/references/analysis-guide.md` | Starting points, tuning, reading results, pitfalls, playbooks, custom features, thresholds, drift signal plots, checklist |
| `skills/kairo-drift-analyst/references/report-template.md` | Entry template for `results.md` |
| `skills/kairo-drift-analyst/scripts/kairo_pipeline.py` | Optional helper: a whole pipeline from a config, run report, sweeps, state profiles, feature shifts, abstractions, figures, the drift signal plot and CSV for the entry and a JSON summary |
| `skills/kairo-drift-analyst/assets/config_template.json` | Config format of the helper |

Using the helper (from the repository root):

```python
import sys; sys.path.insert(0, "skills/kairo-drift-analyst/scripts")
import kairo_pipeline as kp
result = kp.run_pipeline(kp.load_config("analyses/<run folder>/resource.json"))
print(kp.top_windows(result["signals"][0]["report"]))
```

or `uv run python skills/kairo-drift-analyst/scripts/kairo_pipeline.py analyses/<run folder>/resource.json --out analyses/<run folder>/resource`.
