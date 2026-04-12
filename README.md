# Thesis project

## Orientation

This work is **machine-learning oriented**. A relevant reference is [States and SOM](https://www.alessandroberti.it/new_papers/2025_Berti_States_SOM.pdf) (feature extraction, PCA, and SOM). The topic is also **concrete enough for a BSc thesis**: besides theory, you implement a **dashboard** for exploring the inferred states.

---

## Title

**State-based process monitoring in traditional event logs for concept drift detection**

---

## Summary

You study how to describe process behavior using several **complementary notions of state** in a **traditional event log**. The log follows the standard process-mining setting: each event belongs to exactly one case and carries at least a case identifier, an activity, a timestamp, and optionally resource-related attributes.

The thesis is inspired by execution states and boundary conditions from *Identifying Execution States and Boundary Conditions in OCELs*, but the goal is **not** object-centric event logs. You **transfer and adapt** the state-based view to **classical** event logs.

---

## Three kinds of state

### 1. Intra-case state

Describes the **current situation of a single running case**, derived from the sequence of events in that case so far. Possible ingredients include executed activities and their order, time since the previous event, total elapsed time in the case, and relevant case attributes.

### 2. Resource state

For each relevant resource attribute, define a **resource state over the whole event log** that summarizes how that resource dimension behaves **across all cases**. Examples: workload, handover patterns, waiting times, or the distribution of activities for a resource, role, or team.

### 3. Inter-case state

Captures the **global process situation across cases**, based on the sequence of events in the log as a whole and on **temporal gaps** between events. The aim is to capture phases such as normal flow, congestion, bursty activity, or unusually slow behavior.

---

## Core challenge

Make these state notions **computable and interpretable**. States should not only be detected but **explained**. For each type, the analysis should address questions such as:

- What is characteristic for this state compared with others?
- Which conditions are typically observed **before entering** this state?
- Which conditions are associated with **leaving** it?

---

## Implementation goal

Build a **dashboard** for exploring these states. After states are computed, the dashboard should let users:

- Inspect each state and **compare** it with others
- Understand **main characteristics**
- Visualize conditions that often **lead to entering or exiting** a state
- Show **how often each state occurs over time**, so process evolution can be monitored

---

## Evaluation goal

Investigate whether the **combination** of these states is a useful signal for **concept drift detection**: whether significant changes in intra-case, resource, and inter-case states indicate that the underlying process has changed. This can be done by turning states and their frequencies into **time-dependent signals** and checking whether changes align with known or **injected** drift points.

---

## Instance-spanning constraints

These constraints capture dependencies **across multiple cases**, not only within one case. In this thesis they can:

1. Help characterize **inter-case** and **resource-related** states
2. Support **interpretation** of state changes (e.g. cross-case interactions, competition for shared resources, or other global effects)

---

## Research question

**Can a process be monitored through a combination of intra-case, resource, and inter-case states, and is this combination a useful indicator of concept drift in traditional event logs?**

---

## Likely work plan

1. Survey literature on state detection, concept drift detection, and instance-spanning constraints in process mining (breadth over depth).
2. Adapt the notion of execution state from the inspiration paper to **traditional** event logs.
3. Define features for intra-case, resource, and inter-case states.
4. Implement a method to **compute** these states from an event log.
5. Design and implement a **dashboard** for exploring states and transitions.
6. Evaluate whether state-based signals can detect **concept drift**.
