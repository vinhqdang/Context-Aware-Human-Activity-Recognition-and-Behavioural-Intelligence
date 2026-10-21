# Chapter Proposal

Volume: *Context-Aware Human Activity Recognition and Behavioural Intelligence*
Editors: Dr. Singh and Dr. Bijalwan

## Chapter title

When Can an Activity-Recognition Agent Be Trusted to Act? Counterfactual Faithfulness of LLM Explanations and Accountability in Ambient Assisted Living

## Abstract

Large language models (LLMs) are increasingly used as reasoning layers on top of activity recognition systems in smart homes and ambient assisted living (AAL). Beyond labelling activities, such agents explain their conclusions and may trigger actions, such as alerting a caregiver or requesting emergency assistance. Their explanations are fluent, yet recent evaluations show that they can invent correlations with sensor data, which makes explanation quality a question of safety and not only of interpretability. This chapter asks when an activity-recognition agent can justifiably be allowed to act.

We make two contributions. First, we propose a counterfactual faithfulness protocol for LLM explanations over ambient sensor streams. For each explanation that cites specific sensor events as evidence, we intervene on the cited events (removal, value change, temporal shift) and on uncited events, and measure whether the agent's conclusion changes accordingly. An explanation is faithful to the extent that interventions on cited evidence alter the conclusion while interventions on uncited evidence do not. We describe a pilot study on CASAS smart-home data comparing several open and proprietary LLMs under plain and context-enriched prompting, reporting recognition accuracy alongside faithfulness to examine whether the two diverge. Second, we connect the measured faithfulness to accountability. Drawing on automated decision-making and human-oversight rules in EU law (GDPR, the AI Act) and on the Vietnamese regulatory setting, we analyse the evidentiary standard a model-generated inference can meet, the consent of monitored older adults, and the allocation of responsibility among developers, deployers and caregivers for false alarms and missed events. We then propose a graduated authority ladder that links faithfulness thresholds to permitted actions, from logging, to caregiver notification, to irreversible actions that always require human confirmation.

The chapter is aimed at researchers and practitioners who build or regulate AI-assisted monitoring, and it closes with limitations, including that sensitivity to input changes is necessary but not sufficient evidence of sound reasoning.

## Outline

1. Introduction: from activity recognition to agents that explain and act
2. Background: LLM-based reasoning over ambient and wearable sensing, and known failure modes of generated explanations
3. A counterfactual faithfulness protocol for sensor-grounded explanations
4. Pilot study on CASAS: accuracy versus faithfulness across LLMs and prompting strategies
5. From measured faithfulness to accountability: evidence, consent and responsibility in AAL under EU and Vietnamese law
6. A graduated authority ladder for agent actions
7. Limitations and open research directions

## Author and affiliation

Quang-Vinh Dang
British University Vietnam, Hung Yen, Vietnam
Email: vinh.dq4@buv.edu.vn
ORCID: 0000-0002-3877-8024

## Author biography

Quang-Vinh Dang is Senior Lecturer (Associate Professor) at British University Vietnam and Senior Data & AI Director at SmartOSC Vietnam, where he works on enterprise generative and agentic AI. He holds a PhD in Computer Science (Université de Lorraine), an MSc in Financial Engineering (WorldQuant University) and a BSc in Law (Hanoi Law University). His research focuses on artificial intelligence, machine learning and graph methods.
