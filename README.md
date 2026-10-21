# Context-Aware Human Activity Recognition and Behavioural Intelligence

Book chapter: *When Can an Activity-Recognition Agent Be Trusted to Act? Counterfactual Faithfulness of LLM Explanations and Accountability in Ambient Assisted Living.*

## Contents

- `chapter/proposal.md`: the submitted proposal.
- `chapter/chapter.tex`, `refs.bib`, `chapter.pdf`: the chapter (build with `pdflatex`, `bibtex`, `pdflatex` twice).
- `code/faithaal/`: the protocol (`interventions.py`, `protocol.py`, `metrics.py`), prompts and LLM client, reference agents, and the authority ladder (`ladder.py`).
- `code/scripts/`: `run_reference.py`, `run_pilot.py`, `analyze.py`, `make_artifacts.py`.
- `data/adl_noerror/`: CASAS scripted-ADL recordings (Washington State University, CC BY 4.0, Zenodo record 15712834).
- `results/`: protocol records and summaries for the reference agents.

## Reproduce

```
pip install -r requirements.txt
cd code && python -m pytest tests
cd scripts
python run_reference.py --data ../../data/adl_noerror --out ../../results
python analyze.py --records ../../results/records_reference.jsonl --out ../../results --prefix reference
python make_artifacts.py
```

## Status

Done: protocol, reference-agent validation, authority ladder, language-model pilot, chapter text.

Language-model pilot (`results/llm/records_*.jsonl`, summary in `results/summary.csv`): GPT-6-Luna and
DeepSeek-V4.1-Flash on all 120 recordings under both prompts; Laguna-S-2.1 on 20 recordings (both prompts);
gpt-oss-120b, gpt-oss-20b and Qwen3.8-27B on 15 recordings (plain prompt). Runs that produced too few answers
(Ministral-8B, MiniMax-M3.1-Flash, Nemotron-3-Ultra; Gemini unavailable; Groq enriched prompts stalled on daily
limits) are not reported; the Mistral records are kept in `results/llm_excluded/`.

To rerun, export the provider keys as `FAITHAAL_OPENROUTER_KEY`, `FAITHAAL_GROQ_KEY`, `FAITHAAL_MISTRAL_KEY`,
`FAITHAAL_GEMINI_KEY`, then:

```
cd code/scripts
python run_pilot.py --data ../../data/adl_noerror --out ../../results/llm --models openrouter:openai/gpt-6-luna --tag luna
python analyze.py --records ../../results/llm/records_*.jsonl --out ../../results --prefix summary
python make_artifacts.py
```

## Additional experiments

- Free-living segments (`data/home_segments.jsonl`, sampled from the CASAS labelled long-term data, CC BY 4.0, Zenodo record 15708568):
  `python code/scripts/build_home_dataset.py --raw <extracted labeled/ dir> --out data/home_segments.jsonl`, then
  `run_pilot.py --task home ...`; results in `results/home/` and `results/home_summary.csv`.
- Explain-only (a classifier decides, a language model explains): `run_explain.py` stores the cited events in
  `results/explain/`; `analyze_explain.py --task adl|home ...` scores them against the classifier and assigns ladder
  levels (`results/explain_summary_*.csv`). `make_artifacts2.py` builds the chapter tables and figure.
- The free Laguna model stalled on rate limits in both experiments and is excluded.
