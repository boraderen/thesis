# kairo-drift-analyst

Agent skill for blind concept-drift analysis of one event log with kairo. The agent works in its
own run folder in `analyses/` and puts its result into a new numbered result folder of the log,
e.g., `analyses/padella/log_07/1/`, `2/`, … with `results.md`, `plots/` and `drift_signals/`. It
reports exactly one configuration and threshold per required perspective: two (intra-case and
resource) for a Padella log, three (intra-case, resource and inter-case) for every other log.
Agents start from
`SKILL.md`; this file is for humans.

Benchmark notebooks specify a result run and use its configuration and threshold per
perspective directly from `results.md`, without selecting configurations by benchmark score.

## Install

Run from the repository root.

Claude Code:

```bash
mkdir -p .claude/skills && ln -s ../../skills/kairo-drift-analyst .claude/skills/kairo-drift-analyst
```

Codex:

```bash
mkdir -p .agents/skills && ln -s ../../skills/kairo-drift-analyst .agents/skills/kairo-drift-analyst
```

OpenCode:

```bash
mkdir -p .opencode/skills && ln -s ../../skills/kairo-drift-analyst .opencode/skills/kairo-drift-analyst
```

## Use

Start the agent in the repository root with the same prompt. In the apps, paste the prompt into
the chat.

Claude Code:

```bash
claude 'Use the kairo-drift-analyst skill. Task: find the drift points and their perspectives in data/thesis/eval/padella/log_07.xes, Agent: Claude Code, Model: <model>, Features: built-in only'
```

Codex:

```bash
codex 'Use the kairo-drift-analyst skill. Task: find the drift points and their perspectives in data/thesis/eval/padella/log_07.xes, Agent: Codex, Model: <model>, Features: built-in only'
```

OpenCode:

```bash
opencode run 'Use the kairo-drift-analyst skill. Task: find the drift points and their perspectives in data/thesis/eval/padella/log_07.xes, Agent: OpenCode, Model: <model>, Features: built-in only'
```

`Features` is `built-in only` or `custom allowed`. Without installing, start the prompt with
"Read skills/kairo-drift-analyst/SKILL.md completely and follow it." instead.
