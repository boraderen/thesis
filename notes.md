# Thesis notes

Use this file to collect rough ideas, partial thoughts, references to revisit, and questions that are not ready for the main proposal yet.

---

## Work log

### 2026-04-26
- Organized papers into the `papers/` folder
- Read CDLG paper (Grimm, Kraus & van der Aa, BPM 2022)
- Created `logs/generate_drift_log.py` — generates synthetic XES logs with concept drifts (sudden, gradual, recurring, incremental) using the CDLG package
- Created `logs/split_log.py` — reads an event log and a hardcoded list of timestamps, cuts the log into sublogs per time window, exports each as a separate XES file

---

## Current ideas

- For proper benchmarking, synthetic drift logs should cover more than the single **control-flow perspective** currently supported by CDLG; we likely need support for additional perspectives such as resources, data attributes, or timing/arrival behavior.

## Open questions

- **Log splitting for benchmarking** (ask Berti): when cutting a synthetic log at a known drift timestamp for evaluation, what is the right `filter_time_range` mode in pm4py? `traces_contained` drops cross-boundary traces cleanly but loses data; `traces_completing_in` keeps traces intact but places pre-drift events into the post-drift window. Which approach is standard for benchmarking drift detectors, and does it matter for the metrics we care about?
- **CDLG extension vs. new framework vs. PM4Py contribution** (ask Berti): my current preference is to extend the CDLG idea for the thesis with multi-perspective drift generation and gold-standard annotations, while designing it cleanly enough that it could later become a PM4Py contribution and/or paper. Does this direction make sense, or would you rather scope the thesis directly around a new PM4Py log-generation module? Alternative paths are extending CDLG itself and publishing that, building a new framework, or combining some of these if the scope is realistic.

## Notes from reading

### CDLG (Grimm, Kraus & van der Aa, BPM 2022)

- We use CDLG to generate synthetic XES event logs with known concept drifts for experiments
- Four drift types: **sudden** (hard switch), **gradual** (interleaved transition window), **recurring** (v1/v2 alternate seasonally), **incremental** (evolves through multiple intermediate versions)
- Only **control-flow perspective** is supported — cannot model drifts in other perspectives like resource allocation, arrival rates, or data attributes (listed as future work in the paper)
- Gold standard is embedded in `log.attributes['drift info']` — records drift type, exact timestamp, and which activities were added/deleted/moved

## Possible methods

- **Supervised drift classifier over time windows**: train a model on logs labeled with and without concept drift; given a set of time windows extracted from an event log, predict whether each window contains a drift or not. The model learns drift-indicative features from labeled examples and outputs a binary (drift / no-drift) signal per window.
- **Sequential drift detector (LSTM/RNN over state vectors)**: for each case, extract a state vector at every event (encoding the intra-case state at that point in time — e.g., activities seen so far, elapsed time, current resources) and feed the sequence of state vectors into an LSTM or RNN. The model predicts at each step whether a concept drift is currently occurring, learning temporal patterns that distinguish stable from drifting process behavior.
- **Autoencoder / VAE on trace representations**: train on traces from a stable period; reconstruction error or latent-space KL divergence spikes when the incoming behavior no longer matches the learned distribution — fully unsupervised, no drift labels required.
- **GNN over DFGs per time window**: represent each time window as a directly-follows graph and embed it with a GNN; drift is detected as a significant shift in consecutive graph embeddings, capturing control-flow structure without flattening traces into vectors.
- **BERT-style next-activity prediction with confidence monitoring**: pre-train a transformer to predict the next activity in a trace; monitor prediction confidence or loss over time — sustained drops indicate the learned process model no longer fits incoming behavior, signaling drift without explicit labels.
- **Autoencoder as drop-in replacement for PCP compression**: in the original PCP pipeline, replace the hand-crafted compression/feature-extraction step with a learned autoencoder bottleneck; the encoder maps a window's trace set into a compact latent vector, and drift is detected by tracking distances or statistical tests on those latent vectors across windows — same pipeline skeleton, learned representations instead of manual features.

## Dashboard ideas

- Cut an event log with concept drifts at specific timestamps and generate a DFG or Petri net for each sublog — display them side by side to visually compare how the control-flow structure changes across windows
- Extend Oasis with a concept drift detection module: slider UI for cutting the event log at arbitrary timestamps, generating and comparing DFGs/Petri nets per window, and surfacing detected drift points visually

## Evaluation ideas

- Create Petri net of ideal process, simulate a log from it, then inject deviating traces — use as controlled benchmark
- Compare PCP with autoencoders etc.

## To discuss

- 
