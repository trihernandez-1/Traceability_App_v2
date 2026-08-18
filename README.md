# Civic Evidence Studio — functional Streamlit prototype

From raw community-engagement comments to transparent, human-reviewed evidence
connected to planning decisions. Built around the Santa Monica Airport
Conversion Project CoMap housing comments (Phase 3A).

```
RAW ENGAGEMENT FILES → METADATA + CONTEXT → QUANTITATIVE ANALYSIS
→ AI-ASSISTED THEMATIC EXPLORATION → HUMAN REVIEW → EVIDENCE LIBRARY
alongside  PROJECT CONSTRAINTS → CONSTRAINTS LIBRARY
then       EVIDENCE + CONSTRAINTS → DECISION TRAIL
```

AI proposes (clusters, suggested themes, tags, possibly-relevant constraints).
Humans interpret, edit, merge, reject, validate, and decide. Every theme links
back to the exact comments that produced it; every quote keeps its complete
original comment and response ID; every constraint keeps its source.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the app:

1. **01 Data + Context** — the hierarchy is PROJECT → ENGAGEMENT ACTIVITY →
   DATASETS. Review the project card, open the CoMap Scenario Exercise
   activity, and upload all three files (`HOUSING SCENARIO 1/2/3.xlsx`) into
   that ONE activity — one activity can hold many files. Each file gets its
   own dataset metadata (name, scenario, topic); scenario and topic are
   guessed from the filename and editable. Press **Process Activity
   Datasets**. Project constraints (e.g. Measure LC) live at the project
   level in the second tab.
2. **02 Insights Playground** — Overview (reaction counts by scenario),
   Comments (search/filter/tag/save evidence and quotes), Themes (press
   **Run Thematic Analysis** for per-scenario TF-IDF + KMeans clusters with
   AI-suggested names; inspect every underlying comment, including
   counter-evidence; validate into themes), Compare (cross-scenario reaction
   and theme comparison with AI-proposed patterns that require human review).
3. **03 Libraries** — the Evidence Library is organized around THEMES
   (expandable, with interpretation, tags, reaction distribution, datasets
   represented, key evidence, supporting comments, counter-evidence, and
   quantitative patterns). Evidence saved without a theme lands in an
   Unassigned Evidence bucket for later assignment. Constraints Library sits
   alongside. Both support "Add to Decision" staging.
4. **04 Decision Trails** — document a decision and attach evidence,
   constraints, and conflicting input.

No demo data is hard-coded: all counts, clusters, and themes are computed from
whatever files you upload. `sample_data/` contains synthetic files (generated
by `make_sample_data.py`) so you can try the workflow before using the real
CoMap export — replace them with the real files for actual analysis.

## AI configuration (optional)

Set one of these (environment variable or `.streamlit/secrets.toml`):

```
ANTHROPIC_API_KEY=...   # uses claude-sonnet-4-5
# or
OPENAI_API_KEY=...      # uses gpt-4o-mini
```

Without a key, everything still works — upload, counts, comment browsing,
manual tagging, manual themes, libraries, decision trails, and local text
clustering — except AI-generated theme names/summaries, which fall back to
clearly-labeled keyword labels.

## Analytical rules baked in

Reaction is the participant-supplied field, never AI sentiment. Original
comments are never modified; quotes must be exact passages. Theme counts
always come from associated record IDs, never from the LLM. AI output stays
labeled as AI-generated until a human reviews it; validating preserves the
original AI name/summary unchanged alongside the human interpretation.
A recurring theme is not treated as consensus — opposing comments inside a
cluster are surfaced as counter-evidence. Multiple comments can share a
`responseId`, so the app reports both comment counts and unique response-ID
counts. Constraints are documented from authoritative sources; a participant
mentioning Measure LC is evidence about their understanding, not the
constraint itself — AI may suggest a constraint is relevant to a theme, but
only a human can confirm the relationship.

## State

Everything lives in `st.session_state` for this first prototype (survives
navigation, resets on browser refresh). The data structures (evidence items,
themes, constraints, decisions are all plain dicts with stable IDs) are
designed so persistent storage (SQLite/files) can be added later without
reshaping the app.
