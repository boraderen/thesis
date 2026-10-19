# kairo-drift-analyst: hard rules

These rules hold whenever you work with the kairo-drift-analyst skill
(`skills/kairo-drift-analyst/SKILL.md`), also after the conversation was compacted.

- Run Python only as `uv run python -c "..."` or `uv run python <script>.py` from the repository
  root. Bare `python` and `python3` lack kairo's dependencies.
- Never install, upgrade or remove packages. kairo and all its dependencies are installed in the
  `uv` environment.
- Never open an event log (`.xes`, `.csv`) or anything under `data/` with a file-reading tool. Read
  the log in Python with `kairo.read_log`.
- Never open the forbidden files listed in `SKILL.md`.
- Paths of this skill's own files start with `skills/kairo-drift-analyst/`, e.g.,
  `skills/kairo-drift-analyst/assets/config_template.json`.
- Work only in your run folder `analyses/<agent>-<YYYY-MM-DD-HHMM>-<log file name>/`: scripts,
  configs, outputs, figures. Outside it, write only into your own new result folder
  `analyses/<benchmark>/<log name>/<k>/` (`SKILL.md`, *Results files*): your drift signal plots to
  its `plots/`, your drift signal CSVs to its `drift_signals/` and your entry to its `results.md`.
  Other agents may run in parallel: never read, change or delete their run folders or result
  folders.
- Keep your progress in `progress.md` in the run folder: inputs, runs with their configuration,
  findings, next step. Update it after every run.
- After a compaction, first read `SKILL.md` and `progress.md` again, then go on from the next step.
  Do not read the thesis chapters or the kairo docs again; look up only what you need.
- If a command fails, fix it and run it again. Never give up on running Python, and never write
  results instead of computing them.
- Report only results of runs that finished. If no run finished, say so; never claim drifts, or
  that there are none, without a drift signal you computed.
- Report at most one configuration and one threshold per perspective.
- Finish in this order: claim your result folder with the next free number (`kp.new_result_folder`
  or the `mkdir` loop in `SKILL.md`, *Results files*), save the drift signal plots and CSVs into
  it, write your entry to its `results.md` with
  `skills/kairo-drift-analyst/references/report-template.md`, then delete your run folder
  (`rm -r analyses/<your run folder>`) and nothing else.
