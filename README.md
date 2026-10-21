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

Done: protocol, reference-agent validation, authority ladder, chapter text.
Not done: the language-model runs. `run_pilot.py` needs an `OPENROUTER_API_KEY` with budget; after
`python run_pilot.py --data ../../data/adl_noerror --out ../../results` run
`python analyze.py --records ../../results/records.jsonl --out ../../results` and `python make_artifacts.py`
to produce the language-model table and figure. The chapter marks the dependent text as pending.
