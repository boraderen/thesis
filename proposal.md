Thesis proposal:
It is highly ML focused (see the reference paper https://www.alessandroberti.it/new_papers/2025_Berti_States_SOM.pdf in which feature extraction+PCA+SOM have been applied) and at the same time concrete enough for a BSc thesis (as you have also to implement a dashboard exploring the states).
 
State Based Process Monitoring in Traditional Event Logs for Concept Drift Detection
 
In this thesis, you will study how to describe the behavior of a process by means of several complementary notions of state in a traditional event log. The log is assumed to follow the standard process mining setting, where each event belongs to exactly one case and contains at least a case identifier, an activity, a timestamp, and possibly resource related attributes.
 
The thesis is inspired by the idea of execution states and boundary conditions from the paper Identifying Execution States and Boundary Conditions in OCELs, but the goal here is not to work with object centric event logs. Instead, you will transfer and adapt the state based perspective to classical event logs.
 
The main goal is to define and study three types of state.
 
First, an intra case state should describe the current situation of a single running case. This state should be derived from the sequence of events that has happened in that case so far. Possible ingredients are the executed activities, their order, the time since the previous event, the total time elapsed in the case, and relevant case attributes.
 
Second, for each relevant resource attribute, you should define a resource state over the whole event log. This state should summarize how the corresponding resource dimension currently behaves across all cases. Examples are workload, handover patterns, waiting times, or the distribution of activities performed by a certain resource, role, or team.
 
Third, you should define an inter case state that captures the global process situation across cases. This state should be based on the sequence of events in the log as a whole and on temporal gaps between events. The aim is to capture whether the process is currently in a phase such as normal flow, congestion, bursty activity, or unusually slow behavior.
 
A central challenge of the thesis is to make these state notions both computable and interpretable. The states should not only be detected, but also explained. For each type of state, the analysis should answer questions such as: what is characteristic for this state compared with the others, which conditions are typically observed before entering this state, and which conditions are associated with leaving it.
 
The implementation goal is to build a dashboard that supports the exploration of these states. After the states have been computed, the dashboard should allow the user to inspect each state, compare it with other states, and understand its main characteristics. It should also visualize the conditions that often lead to entering or exiting the state. In addition, the dashboard should show how frequently each state occurs over time, so that the evolution of the process can be monitored.
 
The evaluation goal is to investigate whether the combination of these states can serve as a useful signal for concept drift detection. In other words, you will study whether significant changes in the intra case, resource, and inter case states indicate that the underlying process has changed. This can be evaluated by transforming the states and their frequencies into time dependent signals and checking whether changes in these signals correspond to known or injected drift points.
 
A further aspect of the thesis is the connection to instance spanning constraints. These constraints refer to dependencies that involve multiple cases rather than a single case only. In this thesis, they can be used in two ways. First, they can help characterize inter case and resource related states. Second, they can support the interpretation of state changes by showing whether a shift in state is linked to cross case interactions, competition for shared resources, or other global effects.
 
Overall, the thesis should answer the following question: Can a process be monitored through a combination of intra case, resource, and inter case states, and is this combination a useful indicator of concept drift in traditional event logs?
 
The thesis will likely include the following steps:
Study (not too deep) the literature on state detection, concept drift detection, and instance spanning constraints in process mining.
Adapt the notion of execution state from the inspiration paper to the setting of traditional event logs.
Define features for intra case, resource, and inter case states.
Implement a method to compute these states from an event log.
Design and implement a dashboard for exploring the states and their transitions.
Evaluate whether state based signals can detect concept