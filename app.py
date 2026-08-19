"""
CIVIC EVIDENCE STUDIO — functional Streamlit prototype (v3, universal)
======================================================================
Information hierarchy:

  PROJECT
    └── ENGAGEMENT ACTIVITIES        (one activity can contain MANY files)
          └── DATASETS / FILES       (distinguished by a generic DATASET DIMENSION)
                └── RAW RECORDS
                      → ANALYSES     (what the planner wants to learn)
                          → AI-SUGGESTED PATTERNS → HUMAN THEMES → EVIDENCE
                             (+ PROJECT CONSTRAINTS) → DECISION TRAILS

The Insights Playground is not generated directly from uploaded files.
An ANALYSIS defines: which datasets, what each dataset represents
(comparison dimension), the analytical purpose, unit of analysis,
questions to investigate, and which playground modules are enabled —
configured from DATA CAPABILITIES (profiled from the actual data)
plus ANALYSIS GOALS (confirmed by the user; AI only suggests).

Provenance is preserved at every level: each processed record carries
project_id, activity_id, dataset_id, source_file, dim_value, topic,
record_id, response_id, the unaltered original comment, and reaction.
Themes and evidence additionally carry the analysis that produced them.

AI proposes; humans interpret, validate, and decide.
"""

import os
import re
import json
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# PAGE CONFIG + VISUAL SYSTEM
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Civic Evidence Studio", page_icon="🏛️", layout="wide")

C = {
    "bg": "#FAFAF8", "card": "#FFFFFF", "text": "#2E2E2E", "text2": "#6B7280",
    "border": "#E5E7EB",
    "blue": "#8C9AC3", "purple": "#A88BAD", "green": "#81AF98",
    "yellow": "#9BA174", "orange": "#AE9B7F", "red": "#BA9190",
    "blue_t": "#EEF0F6", "purple_t": "#F3EEF4", "green_t": "#EAF2EE",
    "yellow_t": "#F1F1E9", "orange_t": "#F3EEE8", "red_t": "#F3EAEA",
}
REACTION_COLOR = {"approve": C["green"], "disapprove": C["red"], "none": C["yellow"]}

st.markdown(
    f"""
<style>
.stApp {{ background-color: {C['bg']}; }}
html, body, [class*="css"] {{ color: {C['text']}; }}
h1, h2, h3, h4 {{ color: {C['text']}; font-weight: 700; }}
[data-testid="stSidebar"] {{ background-color: {C['card']}; border-right: 1px solid {C['border']}; }}
[data-testid="stMetric"] {{
  background: {C['card']}; border: 1px solid {C['border']}; border-radius: 12px;
  padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}}
[data-testid="stMetricLabel"] p {{ color: {C['text2']}; font-size: 12px;
  text-transform: uppercase; letter-spacing: .5px; font-weight: 700; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: {C['card']}; border-radius: 12px; }}
.stButton > button {{ border-radius: 8px; border: 1px solid {C['border']}; font-weight: 600; }}
.stButton > button[kind="primary"] {{ background: {C['blue']}; border-color: {C['blue']}; color: #fff; }}
.stButton > button[kind="primary"]:hover {{ background: #7e8db8; border-color: #7e8db8; color:#fff; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 18px; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
.ces-pill {{
  display:inline-block; padding:2px 10px; border-radius:100px;
  font-size:11px; font-weight:700; letter-spacing:.2px; margin-right:6px;
  border:1px solid transparent; white-space:nowrap;
}}
.ces-quote {{ font-size:14px; color:{C['text']}; line-height:1.55; margin:2px 0 6px 0; }}
.ces-meta {{ font-size:11.5px; color:{C['text2']}; }}
.ces-note-ai {{ background:{C['purple_t']}; border:1px solid #DDCBE0; border-radius:8px;
  padding:10px 12px; font-size:12.5px; color:#6d5372; margin:6px 0; }}
.ces-note-human {{ background:{C['blue_t']}; border:1px solid #C7CEE0; border-radius:8px;
  padding:10px 12px; font-size:12.5px; color:#3d4661; margin:6px 0; }}
.ces-note-warn {{ background:{C['red_t']}; border:1px solid #DDBCBC; border-radius:8px;
  padding:10px 12px; font-size:12.5px; color:#7c4f4e; margin:6px 0; }}
.ces-note-green {{ background:{C['green_t']}; border:1px solid #BBD9C9; border-radius:8px;
  padding:10px 12px; font-size:12.5px; color:#3f6853; margin:6px 0; }}
.ces-note-yellow {{ background:{C['yellow_t']}; border:1px solid #D3D6BE; border-radius:8px;
  padding:10px 12px; font-size:12.5px; color:#5c6140; margin:6px 0; }}
.ces-chain {{
  font-family: ui-monospace, monospace; font-size:12px; color:{C['text2']};
  background:{C['bg']}; border:1px solid {C['border']}; border-radius:8px;
  padding:10px 14px; white-space:pre; overflow-x:auto;
}}
</style>
""",
    unsafe_allow_html=True,
)


def pill(text, kind):
    styles = {
        "ai":         (C["purple_t"], "#7c5f82", "#DDCBE0"),
        "human":      (C["blue_t"],   "#454f74", "#C7CEE0"),
        "validated":  (C["green_t"],  "#3f6853", "#BBD9C9"),
        "review":     (C["yellow_t"], "#5c6140", "#D3D6BE"),
        "processing": (C["orange_t"], "#6b5c45", "#DBCBB6"),
        "conflict":   (C["red_t"],    "#7c4f4e", "#DDBCBC"),
        "gray":       ("#F3F4F6",     C["text2"], C["border"]),
        "approve":    (C["green_t"],  "#3f6853", "#BBD9C9"),
        "disapprove": (C["red_t"],    "#7c4f4e", "#DDBCBC"),
        "none":       (C["yellow_t"], "#5c6140", "#D3D6BE"),
    }
    bg, fg, bd = styles.get(kind, styles["gray"])
    return (f'<span class="ces-pill" style="background:{bg};color:{fg};'
            f'border-color:{bd};">{text}</span>')


def pills(*items):
    return " ".join(pill(t, k) for t, k in items)


# ----------------------------------------------------------------------------
# SESSION STATE — HIERARCHICAL PROJECT MODEL
# ----------------------------------------------------------------------------

DEFAULT_PROJECT_METADATA = {
    "project_name": "Santa Monica Airport Conversion Project",
    "client": "City of Santa Monica",
    "project_phase": "Phase 3A",
    "project_description": "",
}

DEFAULT_COMAP_ACTIVITY = {
    "activity_name": "CoMap Scenario Exercise",
    "engagement_method": "Interactive Mapping Survey",
    "activity_date": "July 2025",
    "location": "Santa Monica Airport",
    "stakeholder_groups": "Residents, Workers, Business owners, Other participants",
    "participant_count": "",
    "facilitator": "",
    "purpose": "Gather spatial and qualitative reactions to alternative site scenarios.",
    "notes": "",
}

# What distinguishes one dataset from another within an activity.
# "Scenario" is only one possibility — never the assumed default.
DIMENSION_TYPES = [
    "Scenario", "Location / Neighborhood", "Stakeholder Group",
    "Workshop / Session", "Date / Time Period", "Question / Topic",
    "Engagement Round", "Other",
    "None — files should simply be combined",
]
COMBINED_DIMENSION = "None — files should simply be combined"

UNIT_TYPES = ["Comment", "Participant", "Vote", "Ranking", "Map Point",
              "Interview Segment", "Workshop Contribution", "Idea", "Other"]

# Analytical goals, each gated on a data capability (None = always available).
GOAL_DEFS = [
    ("Common themes", "text"),
    ("Differences between groups / scenarios", "multi_group"),
    ("Reasons for approval", "reaction"),
    ("Reasons for disapproval", "reaction"),
    ("Areas of agreement", "text"),
    ("Areas of conflict", "reaction"),
    ("Conditional support", "text"),
    ("Priorities", "rankings"),
    ("Trade-offs", "text"),
    ("Spatial patterns", "coords"),
    ("Demographic differences", "demographics"),
    ("Changes over time", "dates"),
    ("Outlier / minority perspectives", "text"),
    ("Key quotes", "text"),
]

# Playground module registry: id -> (label, required capability, unavailable reason)
MODULE_DEFS = {
    "overview":  ("Overview", None,
                  ""),
    "comments":  ("Comments", "text",
                  "No open-ended text fields exist in the selected datasets."),
    "themes":    ("Themes", "text",
                  "No open-ended text fields exist in the selected datasets."),
    "compare":   ("Compare", "multi_group",
                  "Only one dataset dimension value is available — "
                  "nothing to compare."),
    "theme_map": ("Theme Map", "text",
                  "No open-ended text fields exist in the selected datasets."),
    "map":       ("Map", "coords",
                  "No geographic coordinates exist in the selected datasets."),
    "timeline":  ("Timeline", "dates",
                  "No date fields exist in the selected datasets."),
    "rankings":  ("Rankings", "rankings",
                  "No ranking variables exist in the selected datasets."),
    "stakeholders": ("Stakeholders", "demographics",
                     "No demographic fields exist in the selected datasets."),
}


def init_state():
    ss = st.session_state
    if "project" not in ss:
        ss.project = {
            "project_id": "PROJ-001",
            "metadata": dict(DEFAULT_PROJECT_METADATA),
            "engagement_activities": [
                {"activity_id": "ENG-001",
                 "metadata": dict(DEFAULT_COMAP_ACTIVITY),
                 "dimension": "Scenario",  # dataset dimension for this activity
                 "datasets": [],          # dataset dicts (see add_dataset)
                 "combined": None}        # processed activity dataframe
            ],
        }
    ss.setdefault("activity_seq", 1)
    ss.setdefault("dataset_seq", 0)
    ss.setdefault("constraints", [])          # PROJECT-level constraints
    ss.setdefault("constraint_seq", 0)
    ss.setdefault("analyses", [])             # ANALYSIS objects (see Analysis Setup)
    ss.setdefault("analysis_seq", 0)
    ss.setdefault("active_analysis_id", None)
    ss.setdefault("analysis_draft", None)     # in-progress Analysis Setup wizard
    ss.setdefault("clusters", {})             # cluster_key -> cluster dict
    ss.setdefault("tags", {})                 # record_id -> [{tag, origin}]
    ss.setdefault("themes", [])               # validated + human themes
    ss.setdefault("theme_seq", 0)
    ss.setdefault("evidence", [])             # evidence items (theme optional)
    ss.setdefault("evidence_seq", 0)
    ss.setdefault("decisions", [])
    ss.setdefault("decision_seq", 0)
    ss.setdefault("decision_staged_themes", [])   # theme_ids staged for next decision
    ss.setdefault("validating_cluster", None)
    ss.setdefault("viewing_cluster", {})
    ss.setdefault("cross_reviews", {})


def next_id(seq_key, prefix, width=3):
    st.session_state[seq_key] += 1
    return f"{prefix}{st.session_state[seq_key]:0{width}d}"


def all_activities():
    return st.session_state.project["engagement_activities"]


def get_activity(activity_id):
    return next((a for a in all_activities() if a["activity_id"] == activity_id), None)


def get_dataset(activity, dataset_id):
    return next((d for d in activity["datasets"] if d["dataset_id"] == dataset_id), None)


def get_analysis(analysis_id):
    return next((a for a in st.session_state.analyses
                 if a["analysis_id"] == analysis_id), None)


def active_analysis():
    return get_analysis(st.session_state.active_analysis_id)


def analysis_name(analysis_id):
    a = get_analysis(analysis_id)
    return a["analysis_name"] if a else (analysis_id or "—")


def analysis_df(analysis):
    """Records for one analysis: the activity's processed frame restricted to
    the analysis's datasets, with dim_value set from the analysis's confirmed
    dataset→value mapping (the comparison dimension)."""
    activity = get_activity(analysis["activity_id"])
    if activity is None or activity["combined"] is None:
        return None
    df = activity["combined"]
    df = df[df["dataset_id"].isin(analysis["dataset_ids"])].copy()
    if df.empty:
        return None
    mapping = analysis["comparison_dimension"].get("dataset_values", {})
    if analysis["comparison_dimension"]["name"] == COMBINED_DIMENSION:
        df["dim_value"] = "All records"
    elif mapping:
        df["dim_value"] = df["dataset_id"].map(
            lambda d: mapping.get(d) or "(unspecified)")
    return df


def analysis_frames():
    """All processed activity dataframes, keyed by activity_id."""
    return {a["activity_id"]: a["combined"] for a in all_activities()
            if a["combined"] is not None}


def full_df():
    """Every processed record across all activities (provenance intact)."""
    frames = list(analysis_frames().values())
    return pd.concat(frames, ignore_index=True) if frames else None


def get_record(record_id):
    df = full_df()
    if df is None:
        return None
    rows = df[df["record_id"] == record_id]
    return rows.iloc[0] if len(rows) else None


def records_for(record_ids):
    df = full_df()
    if df is None:
        return pd.DataFrame()
    return df[df["record_id"].isin(record_ids)]


# ----------------------------------------------------------------------------
# DATA STANDARDIZATION (per dataset, provenance preserved)
# ----------------------------------------------------------------------------

def _norm_reaction(v):
    s = str(v).strip().lower()
    if s.startswith("dis") or s in ("down", "thumbsdown", "thumbs down", "-1", "dislike"):
        return "disapprove"
    if "approve" in s or s in ("up", "thumbsup", "thumbs up", "+1", "like", "yes"):
        return "approve"
    return "none"


def guess_dim_value(filename):
    """Guess this file's dimension value from its name (e.g. 'Scenario 2')."""
    m = re.search(r"scenario\s*_?\s*(\d)", filename, re.IGNORECASE)
    if m:
        return f"Scenario {m.group(1)}"
    m = re.search(r"(?:round|session|phase|group)\s*_?\s*(\d)", filename,
                  re.IGNORECASE)
    if m:
        return m.group(0).title()
    return ""


def guess_dimension_type(filenames):
    """Guess the DATASET DIMENSION for a set of files (AI-suggested default)."""
    names = " ".join(filenames).lower()
    if "scenario" in names:
        return "Scenario"
    if any(w in names for w in ("round", "phase")):
        return "Engagement Round"
    if any(w in names for w in ("session", "workshop")):
        return "Workshop / Session"
    if any(w in names for w in ("neighborhood", "downtown", "district")):
        return "Location / Neighborhood"
    return "Other"


def guess_topic(filename):
    for topic in ("Housing", "Mobility", "Ecology", "Water", "Revenue", "Connectivity"):
        if topic.lower() in filename.lower():
            return topic
    return ""


# Column-name signals used by both standardization and profiling.
_TEXT_COLS = ("comment", "comments", "commenttext", "text", "response",
              "openended", "feedback", "answer")
_REACTION_COLS = ("reaction", "reactions", "thumb", "thumbs", "sentiment")
_ID_COLS = ("responseid", "respid", "responseld", "participantid", "userid")
_LAT_COLS = ("lat", "latitude", "y", "ycoord")
_LON_COLS = ("lon", "lng", "long", "longitude", "x", "xcoord")
_DATE_COLS = ("date", "timestamp", "datetime", "created", "submitted")
_RANK_HINTS = ("rank", "ranking", "priority")
_VOTE_HINTS = ("vote", "votes", "choice", "selection")
_DEMO_HINTS = ("age", "gender", "zip", "zipcode", "income", "ethnicity",
               "race", "tenure", "residency", "stakeholder")


def _norm_col(c):
    return re.sub(r"[\s_]+", "", str(c).strip().lower())


def profile_dataset(raw_df):
    """Profile an uploaded file: columns, types, and detected capability fields.
    Only reports what actually exists — capabilities are never fabricated."""
    prof = {"n_rows": int(len(raw_df)), "columns": [],
            "has_text": False, "has_reaction": False, "has_response_id": False,
            "has_coords": False, "has_dates": False, "has_rankings": False,
            "has_votes": False, "has_demographics": False}
    lats, lons = False, False
    for c in raw_df.columns:
        nc = _norm_col(c)
        s = raw_df[c]
        missing = int(s.isna().sum())
        if nc in _TEXT_COLS:
            kind = "text"
            prof["has_text"] = True
        elif nc in _REACTION_COLS or nc == "vote":
            kind = "reaction"
            prof["has_reaction"] = True
        elif nc in _ID_COLS or nc == "response":
            kind = "id"
            prof["has_response_id"] = True
        elif nc in _LAT_COLS:
            kind = "coordinate"
            lats = True
        elif nc in _LON_COLS:
            kind = "coordinate"
            lons = True
        elif nc in _DATE_COLS or "date" in nc:
            kind = "date"
            prof["has_dates"] = True
        elif any(h in nc for h in _RANK_HINTS):
            kind = "ranking"
            prof["has_rankings"] = True
        elif any(h in nc for h in _VOTE_HINTS):
            kind = "vote"
            prof["has_votes"] = True
        elif any(h in nc for h in _DEMO_HINTS):
            kind = "demographic"
            prof["has_demographics"] = True
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        else:
            nunique = s.nunique(dropna=True)
            kind = "categorical" if nunique <= max(20, len(s) // 10) else "text"
        prof["columns"].append({"name": str(c), "normalized": nc, "kind": kind,
                                "dtype": str(s.dtype), "missing": missing})
    prof["has_coords"] = lats and lons
    return prof


def data_capabilities(datasets):
    """Aggregate DATA CAPABILITIES across selected datasets. Derived only from
    profiled fields that actually exist."""
    profs = [d.get("profile") for d in datasets if d.get("profile")]
    caps = {
        "text": any(p["has_text"] for p in profs),
        "reaction": any(p["has_reaction"] for p in profs),
        "response_id": any(p["has_response_id"] for p in profs),
        "multi_group": len(datasets) > 1,
        "coords": any(p["has_coords"] for p in profs),
        "dates": any(p["has_dates"] for p in profs),
        "rankings": any(p["has_rankings"] for p in profs),
        "votes": any(p["has_votes"] for p in profs),
        "demographics": any(p["has_demographics"] for p in profs),
    }
    return caps


CAPABILITY_LABELS = [
    ("text", "Open-ended comments"),
    ("reaction", "Reaction variable"),
    ("response_id", "Response / participant IDs"),
    ("multi_group", "Multiple datasets available for comparison"),
    ("coords", "Spatial coordinates"),
    ("dates", "Date / time fields"),
    ("rankings", "Ranking variables"),
    ("votes", "Votes / selections"),
    ("demographics", "Demographic fields"),
]


def capabilities_html(caps):
    rows = []
    for key, label in CAPABILITY_LABELS:
        if caps.get(key):
            rows.append(f'<span style="color:#3f6853;">✓</span> {label} detected')
        else:
            rows.append(f'<span style="color:#7c4f4e;">✕</span> {label} '
                        'not available')
    return ('<div class="ces-meta" style="line-height:1.9;">'
            + "<br>".join(rows) + "</div>")


def standardize_dataset(raw_df, dataset, activity):
    """Normalize an uploaded file into standard records with full provenance.
    Returns (df, problems). Original comments are never altered.
    Comment text is required; reaction and response ID are optional — when a
    field is absent it is recorded as absent, never fabricated."""
    problems = []
    df = raw_df.copy()
    df.columns = [_norm_col(c) for c in df.columns]
    colmap = {}
    for c in df.columns:
        if c in _TEXT_COLS and "comment" not in colmap.values():
            colmap[c] = "comment"
        elif c in _REACTION_COLS or c == "vote":
            colmap[c] = "reaction"
        elif c in _ID_COLS or c == "response":
            colmap[c] = "response_id"
        elif c in _LAT_COLS:
            colmap[c] = "lat"
        elif c in _LON_COLS:
            colmap[c] = "lon"
        elif c in _DATE_COLS:
            colmap[c] = "record_date"
    df = df.rename(columns=colmap)
    if "comment" not in df.columns:
        problems.append(
            f"**{dataset['source_file']}** has no open-ended text column "
            f"(looked for: {', '.join(_TEXT_COLS[:4])}…). Columns found: "
            f"{', '.join(df.columns)}. The app will not fabricate this field.")
        return None, problems

    df["comment"] = df["comment"].astype(str)
    if "response_id" in df.columns:
        df["response_id"] = df["response_id"].astype(str).str.strip()
        has_response = True
    else:
        has_response = False
    if "reaction" in df.columns:
        df["reaction_original"] = df["reaction"]
        df["reaction"] = df["reaction"].apply(_norm_reaction)
    else:
        df["reaction_original"] = ""
        df["reaction"] = "none"
    df = df.reset_index(drop=True)

    num = dataset["dataset_id"].split("-")[-1]
    df["record_id"] = [f"D{num}-{i + 1:05d}" for i in range(len(df))]
    if not has_response:
        # No participant ID exists — each record stands alone. Recorded, not
        # fabricated: unique-participant counts will equal record counts.
        df["response_id"] = df["record_id"]
        problems = []  # not an error, but surface a note on the dataset
        dataset["notes_auto"] = ("No response/participant ID column found — "
                                 "unique participant counts fall back to "
                                 "record counts.")
    df["project_id"] = st.session_state.project["project_id"]
    df["activity_id"] = activity["activity_id"]
    df["dataset_id"] = dataset["dataset_id"]
    df["source_file"] = dataset["source_file"]
    df["dim_value"] = dataset["dim_value"] or "(unspecified)"
    df["topic"] = dataset["topic"] or ""

    cols = ["project_id", "activity_id", "dataset_id", "source_file", "dim_value",
            "topic", "record_id", "response_id", "comment", "reaction",
            "reaction_original"]
    # Carry every other detected column through (coordinates, dates, rankings,
    # demographics…) so capability-gated modules can use the real fields.
    for extra in df.columns:
        if extra not in cols:
            cols.append(extra)
    return df[cols], []


def reaction_counts(df):
    vc = df["reaction"].value_counts()
    return {r: int(vc.get(r, 0)) for r in ("approve", "disapprove", "none")}


# ----------------------------------------------------------------------------
# LLM INTEGRATION (optional — graceful fallback)
# ----------------------------------------------------------------------------

def _get_secret(name):
    v = os.environ.get(name)
    if v:
        return v
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def llm_provider():
    k = _get_secret("ANTHROPIC_API_KEY")
    if k:
        return "anthropic", k
    k = _get_secret("OPENAI_API_KEY")
    if k:
        return "openai", k
    return None, None


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def llm_json(prompt, max_tokens=600):
    """One LLM call returning parsed JSON, or None (no key / error)."""
    provider, key = llm_provider()
    if provider is None:
        return None
    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model="claude-sonnet-4-5", max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            text = msg.content[0].text
        else:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini", max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            text = resp.choices[0].message.content
        return _extract_json(text)
    except Exception as e:
        st.session_state.setdefault("llm_errors", []).append(str(e))
        return None


def llm_interpret_cluster(sample_comments, keywords, scope_label):
    numbered = "\n".join(f"{i+1}. {c[:400]}" for i, c in enumerate(sample_comments[:12]))
    prompt = (
        "You are helping a city planner label a cluster of public-engagement comments "
        f"({scope_label}). These are a representative SAMPLE from the cluster.\n\n"
        f"Top cluster keywords: {', '.join(keywords[:10])}\n\n"
        f"Sample comments:\n{numbered}\n\n"
        "Respond with ONLY a JSON object:\n"
        '{"name": "<SHORT theme name, 2-4 words, e.g. Housing Opposition>", '
        '"summary": "<1-2 sentence neutral description of what these comments express>", '
        '"tags": ["<tag1>", "<tag2>", "<tag3>"]}\n'
        "Keep the name short (2-4 words). Do not overstate consensus."
    )
    data = llm_json(prompt, max_tokens=400)
    if data and data.get("name"):
        return {"name": str(data.get("name", "")).strip(),
                "summary": str(data.get("summary", "")).strip(),
                "tags": [str(t).strip() for t in (data.get("tags") or [])][:5]}
    return None


# ----------------------------------------------------------------------------
# ANALYSIS SETUP SUGGESTIONS (AI proposes; humans confirm)
# ----------------------------------------------------------------------------

def heuristic_analysis_suggestions(activity, datasets, caps):
    """Deterministic fallback suggestions when no LLM is configured.
    Derived only from metadata and profiled fields."""
    md = activity["metadata"]
    files = [d["source_file"] for d in datasets]
    dim = activity.get("dimension") or guess_dimension_type(files)
    values = {d["dataset_id"]: (d["dim_value"] or d["dataset_name"]
                                or d["source_file"]) for d in datasets}
    found = []
    if caps["text"]:
        found.append("written comments")
    if caps["reaction"]:
        found.append("approve / disapprove / none reactions")
    if caps["response_id"]:
        found.append("response IDs")
    if caps["coords"]:
        found.append("geographic coordinates")
    if caps["dates"]:
        found.append("date fields")
    intro = (f"I found {len(datasets)} dataset"
             f"{'s' if len(datasets) != 1 else ''} within "
             f"{md['activity_name'] or 'this activity'}.")
    if found:
        intro += ("\n\nThey contain:\n"
                  + "\n".join(f"- {f}" for f in found))
    if len(datasets) > 1:
        intro += (f"\n\nThe files appear to represent different "
                  f"{dim.lower()}s. Would you like to compare them?")
    purpose = md["purpose"].strip() or (
        "Understand what participants expressed during "
        f"{md['activity_name'] or 'this engagement activity'} and the reasons "
        "behind their responses.")
    goals = []
    if caps["text"]:
        goals.append("Common themes")
        goals.append("Key quotes")
        goals.append("Conditional support")
    if caps["multi_group"]:
        goals.append("Differences between groups / scenarios")
    if caps["reaction"]:
        goals += ["Reasons for approval", "Reasons for disapproval",
                  "Areas of conflict"]
    unit = "Comment" if caps["text"] else ("Vote" if caps["votes"] else "Other")
    topic = (datasets[0]["topic"] or "the proposal") if datasets else "the proposal"
    dl = dim.lower()
    questions = []
    if caps["reaction"]:
        questions.append(f"What reasons drive opposition to {topic.lower()}?")
        questions.append("How do reasons for approval differ from reasons "
                         "for disapproval?")
    if caps["text"]:
        questions.append(f"Are there forms of {topic.lower()} that receive "
                         "conditional support?")
    if caps["multi_group"]:
        questions.append(f"Which themes appear across all {dl}s?")
        questions.append(f"Which concerns are specific to one {dl}?")
    # suggest only constraints participants actually mention in the data —
    # a mention is evidence about their understanding, not the constraint
    # itself; the user confirms relevance.
    texts = " ".join(d["df"]["comment"].str.lower().str.cat(sep=" ")
                     for d in datasets if d.get("df") is not None)
    con_names = [c["name"] for c in st.session_state.constraints
                 if c["name"].strip() and c["name"].strip().lower() in texts]
    return {"intro": intro, "dimension": dim, "dataset_values": values,
            "purpose": purpose, "goals": goals, "unit": unit,
            "questions": questions[:5], "constraints": con_names,
            "source": "heuristic"}


def ai_analysis_suggestions(activity, datasets, caps):
    """AI-suggested Analysis Brief. LLM when configured, heuristics otherwise.
    Suggestions only — every field is confirmed or edited by the user."""
    base = heuristic_analysis_suggestions(activity, datasets, caps)
    provider, _ = llm_provider()
    if provider is None:
        return base
    md = activity["metadata"]
    ds_desc = "\n".join(
        f'- {d["source_file"]} (dataset {d["dataset_id"]}, '
        f'{d["profile"]["n_rows"] if d.get("profile") else "?"} rows; columns: '
        f'{", ".join(c["name"] for c in (d.get("profile") or {}).get("columns", []))})'
        for d in datasets)
    caps_desc = ", ".join(k for k, v in caps.items() if v)
    cons_desc = "; ".join(f'{c["id"]} {c["name"]}'
                          for c in st.session_state.constraints) or "none"
    prompt = (
        "You are helping a city planner set up an ANALYSIS of public-engagement "
        "data. Inspect the context and suggest a draft Analysis Brief. The "
        "planner will confirm or edit everything — make suggestions, do not "
        "overstate.\n\n"
        f"ENGAGEMENT ACTIVITY: {md['activity_name']} — method: "
        f"{md['engagement_method']}; date: {md['activity_date']}; purpose: "
        f"{md['purpose']}\n"
        f"DATASETS:\n{ds_desc}\n"
        f"DATA CAPABILITIES (detected from actual fields): {caps_desc}\n"
        f"DIMENSION TYPES to choose from: {', '.join(DIMENSION_TYPES)}\n"
        f"PROJECT CONSTRAINTS on file: {cons_desc}\n\n"
        "Respond with ONLY a JSON object:\n"
        '{"intro": "<2-4 sentence conversational summary of what you found '
        'and whether comparison makes sense>", '
        '"dimension": "<one of the dimension types — what distinguishes these '
        'datasets from each other>", '
        '"dataset_values": {"<dataset_id>": "<short dimension value>"}, '
        '"purpose": "<1-2 sentence analysis purpose>", '
        '"goals": ["<goal>", "..."], '
        '"unit": "<primary unit of analysis, e.g. Comment>", '
        '"questions": ["<question 1>", "..."], '
        '"constraints": ["<constraint name possibly relevant>"]}\n'
        "Suggest only goals supported by the detected capabilities. Suggest "
        "3-5 questions. Do not suggest spatial analysis unless coordinates "
        "exist.")
    data = llm_json(prompt, max_tokens=900)
    if not data:
        return base
    out = dict(base)
    out["source"] = "llm"
    for k in ("intro", "purpose", "unit", "dimension"):
        if isinstance(data.get(k), str) and data[k].strip():
            out[k] = data[k].strip()
    if out["dimension"] not in DIMENSION_TYPES:
        out["dimension"] = base["dimension"]
    if isinstance(data.get("dataset_values"), dict):
        for did, v in data["dataset_values"].items():
            if did in out["dataset_values"] and str(v).strip():
                out["dataset_values"][did] = str(v).strip()
    valid_goals = {g for g, _ in GOAL_DEFS}
    if isinstance(data.get("goals"), list):
        goals = [str(g).strip() for g in data["goals"] if str(g).strip()]
        goals = [g for g in goals if g in valid_goals] or base["goals"]
        out["goals"] = goals
    if isinstance(data.get("questions"), list) and data["questions"]:
        out["questions"] = [str(q).strip() for q in data["questions"]][:5]
    if isinstance(data.get("constraints"), list):
        known = {c["name"] for c in st.session_state.constraints}
        out["constraints"] = [str(c).strip() for c in data["constraints"]
                              if str(c).strip() in known]
    return out


# ----------------------------------------------------------------------------
# THEMATIC CLUSTERING (local, transparent: TF-IDF + KMeans)
# ----------------------------------------------------------------------------

def cluster_one_group(sdf):
    """Cluster comments for one group (one dimension value, e.g. one scenario
    or one neighborhood). Calculated fields only — AI interpretation attached
    separately."""
    sdf = sdf[sdf["comment"].str.strip().astype(bool)]
    sdf = sdf[~sdf["comment"].str.strip().str.lower().isin(["nan", "none", ""])]
    n = len(sdf)
    if n == 0:
        return []
    texts = sdf["comment"].tolist()
    k = 1 if n < 6 else int(min(5, max(2, round(n / 10))))
    vec = TfidfVectorizer(stop_words="english", max_features=2500,
                          ngram_range=(1, 2), min_df=1)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return []
    terms = np.array(vec.get_feature_names_out())
    if k == 1 or X.shape[0] <= k:
        labels = np.zeros(n, dtype=int)
        centers = np.asarray(X.mean(axis=0))
        k = 1
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_

    clusters = []
    for ci in range(k):
        idx = np.where(labels == ci)[0]
        if len(idx) == 0:
            continue
        sub = sdf.iloc[idx]
        center = np.asarray(centers[ci]).ravel()
        top_terms = terms[np.argsort(center)[::-1][:8]].tolist()
        sims = np.asarray(X[idx].dot(center)).ravel()
        order = np.argsort(sims)[::-1]
        rep_ids = sub.iloc[order[:5]]["record_id"].tolist()
        # similarity to the cluster center, normalized per cluster → a
        # human-review signal: low-confidence members are ambiguous, not wrong
        smax = float(sims.max()) if len(sims) and float(sims.max()) > 0 else 1.0
        confidence = {rid: round(float(s) / smax, 3)
                      for rid, s in zip(sub["record_id"], sims)}
        counts = reaction_counts(sub)
        if counts["approve"] >= counts["disapprove"]:
            majority, minority = "approve", "disapprove"
        else:
            majority, minority = "disapprove", "approve"
        clusters.append({
            "group": sub["dim_value"].iloc[0],
            "record_ids": sub["record_id"].tolist(),
            "dataset_ids": sorted(sub["dataset_id"].unique().tolist()),
            "activity_ids": sorted(sub["activity_id"].unique().tolist()),
            "source_files": sorted(sub["source_file"].unique().tolist()),
            "n_comments": int(len(sub)),
            "n_respondents": int(sub["response_id"].nunique()),
            "counts": counts, "majority": majority,
            "counter_ids": sub[sub["reaction"] == minority]["record_id"].tolist(),
            "keywords": top_terms, "rep_ids": rep_ids,
            "confidence": confidence,
        })
    clusters.sort(key=lambda c: -c["n_comments"])
    return clusters


def analysis_clusters(analysis_id, statuses=("ai-suggested", "validated")):
    """Clusters belonging to one analysis (each analysis keeps its own)."""
    return {k: c for k, c in st.session_state.clusters.items()
            if c.get("analysis_id") == analysis_id and c["status"] in statuses}


def run_thematic_analysis(analysis, scope_df, scope_desc):
    """Cluster the scoped records for ONE analysis, grouped by the analysis's
    comparison dimension; attach AI interpretation. Dataset/activity provenance
    is preserved inside every cluster; clusters carry their analysis of origin."""
    provider, _ = llm_provider()
    aid = analysis["analysis_id"]
    # replace only this analysis's previous clusters — other analyses keep theirs
    st.session_state.clusters = {
        k: c for k, c in st.session_state.clusters.items()
        if c.get("analysis_id") != aid}
    groups = sorted(scope_df["dim_value"].unique())
    for grp in groups:
        sdf = scope_df[scope_df["dim_value"] == grp]
        found = cluster_one_group(sdf)
        for j, cl in enumerate(found):
            key = f"{aid}-{re.sub(r'[^A-Za-z0-9]', '', grp)}-C{j + 1}"
            cl["key"] = key
            cl["analysis_id"] = aid
            cl["status"] = "ai-suggested"
            cl["constraint_links"] = {}
            rep_comments = records_for(cl["rep_ids"])["comment"].tolist()
            ai = llm_interpret_cluster(rep_comments, cl["keywords"],
                                       f"{scope_desc}, {grp}") if provider else None
            if ai:
                cl["ai"] = {**ai, "source": "llm"}
            else:
                kw = [w.title() for w in cl["keywords"][:3]]
                cl["ai"] = {
                    "name": " / ".join(kw) if kw else "Unlabeled cluster",
                    "summary": ("Keyword-based label. AI theme interpretation "
                                "unavailable — configure an API key to enable "
                                "AI-generated theme names and summaries."),
                    "tags": [w.title() for w in cl["keywords"][:3]],
                    "source": "fallback"}
            st.session_state.clusters[key] = cl
    analysis["cluster_run_done"] = True
    analysis["cluster_scope_desc"] = scope_desc


# ----------------------------------------------------------------------------
# TAGS / EVIDENCE / THEME HELPERS
# ----------------------------------------------------------------------------

def add_tag(record_id, tag, origin):
    tag = tag.strip()
    if not tag:
        return
    entry = st.session_state.tags.setdefault(record_id, [])
    if not any(t["tag"].lower() == tag.lower() and t["origin"] == origin for t in entry):
        entry.append({"tag": tag, "origin": origin})


def all_known_tags():
    tags = set()
    for entries in st.session_state.tags.values():
        for t in entries:
            tags.add(t["tag"])
    for cl in st.session_state.clusters.values():
        for t in cl["ai"].get("tags", []):
            tags.add(t)
    for th in st.session_state.themes:
        for t in th.get("tags", []):
            tags.add(t)
    return sorted(tags)


def get_theme(theme_id):
    return next((t for t in st.session_state.themes if t["theme_id"] == theme_id), None)


def add_evidence(item):
    """Create an evidence item. theme_id may be None → Unassigned Evidence.
    Evidence created inside a playground carries its analysis of origin."""
    item["evidence_id"] = next_id("evidence_seq", "EV-", width=4)
    item["created"] = datetime.date.today().isoformat()
    item.setdefault("theme_id", None)
    item.setdefault("analysis_id", st.session_state.active_analysis_id)
    st.session_state.evidence.append(item)
    return item["evidence_id"]


def provenance_from_records(record_ids):
    sub = records_for(record_ids)
    if sub.empty:
        return {"activity_ids": [], "dataset_ids": [], "dim_values": [],
                "source_files": [], "response_ids": []}
    return {
        "activity_ids": sorted(sub["activity_id"].unique().tolist()),
        "dataset_ids": sorted(sub["dataset_id"].unique().tolist()),
        "dim_values": sorted(sub["dim_value"].unique().tolist()),
        "source_files": sorted(sub["source_file"].unique().tolist()),
        "response_ids": sorted(sub["response_id"].unique().tolist()),
    }


def activity_name(activity_id):
    a = get_activity(activity_id)
    return a["metadata"]["activity_name"] if a else activity_id


def comment_card(row, key_prefix, show_actions=True):
    """One comment card with full provenance + actions."""
    with st.container(border=True):
        st.markdown(f'<div class="ces-quote">&ldquo;{row["comment"]}&rdquo;</div>',
                    unsafe_allow_html=True)
        tag_html = ""
        for t in st.session_state.tags.get(row["record_id"], []):
            tag_html += pill(t["tag"], "ai" if t["origin"] == "ai" else "human")
        st.markdown(
            pills((row["reaction"].title(), row["reaction"]),
                  (row["dim_value"], "gray")) + tag_html
            + f'<div class="ces-meta" style="margin-top:6px;">'
              f'{row["record_id"]} · Response {row["response_id"]} · '
              f'{row["source_file"]} · {activity_name(row["activity_id"])}</div>',
            unsafe_allow_html=True)
        if not show_actions:
            return
        theme_opts = {t["theme_id"]: t["name"] for t in st.session_state.themes}
        c1, c2, c3 = st.columns([1.3, 1, 1.4])
        with c1:
            with st.popover("Add to Evidence"):
                sel_theme = st.selectbox(
                    "Assign to theme (optional)",
                    ["Unassigned"] + list(theme_opts),
                    format_func=lambda k: theme_opts.get(k, k),
                    key=f"{key_prefix}-evtheme-{row['record_id']}")
                if st.button("Save as evidence",
                             key=f"{key_prefix}-ev-{row['record_id']}"):
                    add_evidence({
                        "type": "Direct Comment",
                        "record_ids": [row["record_id"]],
                        "dim_value": row["dim_value"],
                        "reaction": row["reaction"],
                        "original_comment": row["comment"],
                        "selected_quote": None,
                        "tags": [t["tag"] for t in
                                 st.session_state.tags.get(row["record_id"], [])],
                        "theme_id": None if sel_theme == "Unassigned" else sel_theme,
                        "status": "Human Selected",
                        **provenance_from_records([row["record_id"]]),
                    })
                    st.toast(f"Saved {row['record_id']} to Evidence Library")
        with c2:
            with st.popover("Add Tag"):
                existing = all_known_tags()
                pick = st.selectbox("Existing tag", ["—"] + existing,
                                    key=f"{key_prefix}-tagsel-{row['record_id']}")
                new = st.text_input("Or new tag",
                                    key=f"{key_prefix}-tagnew-{row['record_id']}")
                if st.button("Apply tag",
                             key=f"{key_prefix}-tagapply-{row['record_id']}"):
                    chosen = new.strip() or (pick if pick != "—" else "")
                    if chosen:
                        add_tag(row["record_id"], chosen, "human")
                        st.rerun()
        with c3:
            with st.popover("Save Quote"):
                st.caption("Select the passage to save. The complete original comment "
                           "and response ID are always preserved alongside it.")
                q = st.text_area("Quoted passage", value=row["comment"],
                                 key=f"{key_prefix}-quote-{row['record_id']}")
                sel_theme_q = st.selectbox(
                    "Assign to theme (optional)",
                    ["Unassigned"] + list(theme_opts),
                    format_func=lambda k: theme_opts.get(k, k),
                    key=f"{key_prefix}-qtheme-{row['record_id']}")
                if st.button("Save highlighted quote",
                             key=f"{key_prefix}-quotesave-{row['record_id']}"):
                    if q.strip() and q.strip() in row["comment"]:
                        add_evidence({
                            "type": "Highlighted Quote",
                            "record_ids": [row["record_id"]],
                            "dim_value": row["dim_value"],
                            "reaction": row["reaction"],
                            "original_comment": row["comment"],
                            "selected_quote": q.strip(),
                            "tags": [t["tag"] for t in
                                     st.session_state.tags.get(row["record_id"], [])],
                            "theme_id": None if sel_theme_q == "Unassigned" else sel_theme_q,
                            "status": "Human Selected",
                            **provenance_from_records([row["record_id"]]),
                        })
                        st.toast("Quote saved with full provenance")
                    else:
                        st.error("The quote must be an exact passage from the "
                                 "original comment — quotes are never fabricated "
                                 "or altered.")


# ----------------------------------------------------------------------------
# PAGE 01 — DATA + CONTEXT
# ----------------------------------------------------------------------------

CONSTRAINT_TYPES = ["Legal / Regulatory", "Voter Mandate", "Financial", "Environmental",
                    "Site / Physical", "Technical", "Timeline", "Other"]


def render_dataset_row(activity, ds):
    """LEVEL 3 — one dataset/file inside an engagement activity."""
    dim_label = activity.get("dimension") or "Distinguishing value"
    if dim_label == COMBINED_DIMENSION:
        dim_label = "Distinguishing value"
    n = len(ds["df"]) if ds["df"] is not None else 0
    label = f'{ds["dataset_name"] or ds["source_file"]} — {n} comments' \
            + ("" if ds["include"] else "  (excluded from analysis)")
    with st.expander(label):
        st.markdown(
            pills((ds["dataset_id"], "gray"),
                  (ds["dim_value"] or f"no {dim_label.lower()}", "gray"),
                  (ds["topic"] or "no topic", "gray"),
                  ("Included" if ds["include"] else "Excluded",
                   "validated" if ds["include"] else "review")),
            unsafe_allow_html=True)
        st.markdown(f'<div class="ces-meta">Source: {ds["source_file"]} · '
                    f'{n} comments · '
                    f'{ds["df"]["response_id"].nunique() if ds["df"] is not None else 0} '
                    f'unique response IDs</div>', unsafe_allow_html=True)
        if ds.get("notes_auto"):
            st.markdown(f'<div class="ces-note-yellow">{ds["notes_auto"]}</div>',
                        unsafe_allow_html=True)
        with st.form(f"dsmeta-{ds['dataset_id']}"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Dataset Name", ds["dataset_name"],
                                     placeholder="e.g. Housing — Scenario 1")
                dim_value = st.text_input(
                    f"{dim_label} (what this file represents)", ds["dim_value"],
                    placeholder="e.g. Scenario 1, Downtown, Session A")
            with c2:
                topic = st.text_input("Topic / Category", ds["topic"],
                                      placeholder="e.g. Housing")
                include = st.checkbox("Include in analysis", value=ds["include"])
            desc = st.text_area("Dataset Description (optional)", ds["description"],
                                height=68)
            notes = st.text_area("Dataset Notes (optional)", ds["notes"], height=68)
            if st.form_submit_button("Save Dataset Metadata", type="primary"):
                ds.update({"dataset_name": name.strip(), "dim_value": dim_value.strip(),
                           "topic": topic.strip(), "description": desc.strip(),
                           "notes": notes.strip(), "include": include})
                activity["combined"] = None  # metadata changed → reprocess
                st.rerun()
        if ds.get("profile"):
            with st.expander("Data profile (detected fields)"):
                prof = ds["profile"]
                st.dataframe(pd.DataFrame(prof["columns"])[
                    ["name", "kind", "dtype", "missing"]],
                    width="stretch", hide_index=True)
                st.caption(f'{prof["n_rows"]} rows. Field kinds are detected '
                           'from the actual data — capabilities are never '
                           'fabricated.')
        if ds["df"] is not None:
            with st.expander("Preview data"):
                st.dataframe(ds["df"].head(10), width="stretch",
                             hide_index=True)
        for p in ds.get("problems", []):
            st.error(p)
        if st.button("Remove Dataset", key=f"rmds-{ds['dataset_id']}"):
            activity["datasets"] = [d for d in activity["datasets"]
                                    if d["dataset_id"] != ds["dataset_id"]]
            activity["combined"] = None
            st.rerun()


def render_activity_card(activity):
    """LEVEL 2 — one engagement activity containing one or more datasets."""
    md = activity["metadata"]
    n_ds = len(activity["datasets"])
    header = (f'{md["activity_name"] or "Untitled Activity"} — '
              f'{md["engagement_method"] or "method not set"} · '
              f'{md["activity_date"] or "date not set"} · '
              f'{n_ds} dataset{"s" if n_ds != 1 else ""}')
    with st.expander(header, expanded=True):
        st.markdown(pills((activity["activity_id"], "gray")), unsafe_allow_html=True)

        st.markdown("**Activity Metadata** — applies to the whole activity "
                    "(when / where / how / who participated). Not repeated per file.")
        with st.form(f"actmeta-{activity['activity_id']}"):
            c1, c2 = st.columns(2)
            with c1:
                a_name = st.text_input("Activity Name", md["activity_name"])
                a_method = st.text_input("Engagement Method", md["engagement_method"])
                a_date = st.text_input("Activity Date or Date Range", md["activity_date"])
                a_loc = st.text_input("Neighborhood / Location", md["location"])
            with c2:
                a_stake = st.text_input("Stakeholder Groups", md["stakeholder_groups"])
                a_count = st.text_input("Number of Participants (optional)",
                                        md["participant_count"])
                a_fac = st.text_input("Facilitator (optional)", md["facilitator"])
                a_purpose = st.text_area("Activity Purpose", md["purpose"], height=68)
            a_notes = st.text_area("Notes (optional)", md["notes"], height=60)
            cur_dim = activity.get("dimension") or "Other"
            a_dim = st.selectbox(
                "Dataset Dimension — what distinguishes one file from another "
                "within this activity?", DIMENSION_TYPES,
                index=DIMENSION_TYPES.index(cur_dim)
                if cur_dim in DIMENSION_TYPES else DIMENSION_TYPES.index("Other"))
            if st.form_submit_button("Save Activity Metadata", type="primary"):
                activity["metadata"] = {
                    "activity_name": a_name, "engagement_method": a_method,
                    "activity_date": a_date, "location": a_loc,
                    "stakeholder_groups": a_stake, "participant_count": a_count,
                    "facilitator": a_fac, "purpose": a_purpose, "notes": a_notes}
                activity["dimension"] = a_dim
                st.rerun()

        st.markdown("---")
        st.markdown("**Datasets / Files** — one activity can contain many files. "
                    "File metadata only describes which subset of the activity the "
                    "dataset represents.")

        ups = st.file_uploader(
            "Add file(s) to this activity (XLSX — an open-ended text column is "
            "required; reaction, response ID, coordinates, and dates are "
            "detected when present)", type=["xlsx"], accept_multiple_files=True,
            key=f"up-{activity['activity_id']}")
        known_files = {d["source_file"] for d in activity["datasets"]}
        added_new = False
        for up in ups or []:
            if up.name in known_files:
                continue
            ds_id = next_id("dataset_seq", "DATA-")
            ds = {"dataset_id": ds_id,
                  "dataset_name": (f'{guess_topic(up.name) or "Dataset"} — '
                                   f'{guess_dim_value(up.name)}').strip(" —"),
                  "source_file": up.name,
                  "dim_value": guess_dim_value(up.name),
                  "topic": guess_topic(up.name),
                  "description": "", "notes": "", "include": True,
                  "df": None, "problems": [], "profile": None}
            try:
                raw_df = pd.read_excel(up)
                ds["profile"] = profile_dataset(raw_df)
                sdf, problems = standardize_dataset(raw_df, ds, activity)
                ds["df"] = sdf
                ds["problems"] = problems
            except Exception as e:
                ds["problems"] = [f"Could not read this file as XLSX: {e}"]
            activity["datasets"].append(ds)
            activity["combined"] = None
            known_files.add(up.name)
            added_new = True
        if added_new:
            st.rerun()

        for ds in activity["datasets"]:
            render_dataset_row(activity, ds)

        included = [d for d in activity["datasets"]
                    if d["include"] and d["df"] is not None]
        st.markdown("---")
        if st.button("Process Activity Datasets", type="primary",
                     key=f"proc-{activity['activity_id']}",
                     disabled=(len(included) == 0)):
            with st.spinner("Combining datasets — provenance preserved…"):
                frames = []
                for d in included:
                    f = d["df"].copy()
                    f["dim_value"] = d["dim_value"] or "(unspecified)"
                    f["topic"] = d["topic"] or ""
                    f["source_file"] = d["source_file"]
                    frames.append(f)
                activity["combined"] = pd.concat(frames, ignore_index=True)
            st.rerun()

        if activity["combined"] is not None:
            df = activity["combined"]
            cts = reaction_counts(df)
            st.markdown(pill("Activity dataset ready", "validated"),
                        unsafe_allow_html=True)
            m = st.columns(4)
            m[0].metric("Combined comments", len(df))
            m[1].metric("Unique response IDs", df["response_id"].nunique())
            m[2].metric("Datasets included", len(included))
            m[3].metric("Approve / Disapprove / None",
                        f'{cts["approve"]} / {cts["disapprove"]} / {cts["none"]}')
            per = df.groupby(["dataset_id", "source_file", "dim_value"]).agg(
                comments=("record_id", "count"),
                unique_respondents=("response_id", "nunique")).reset_index()
            per = per.rename(columns={
                "dim_value": activity.get("dimension") or "Value"})
            st.dataframe(per, width="stretch", hide_index=True)
            st.markdown("**Data Capabilities** — detected from the actual "
                        "fields in the included datasets:")
            st.markdown(capabilities_html(data_capabilities(included)),
                        unsafe_allow_html=True)
            st.caption("Every row retains project_id, activity_id, dataset_id, "
                       "source_file, dimension value, topic, record_id, and "
                       "response_id — relationships are never flattened away. "
                       "Next step: define an Analysis in **02 Analysis Setup**.")


def page_data_context():
    st.title("Data + Context")
    st.markdown(f'<p style="color:{C["text2"]};margin-top:-8px;">Project → Engagement '
                'Activities → Datasets. Document the project, its engagement '
                'activities, the files within each activity, and the project '
                'constraints.</p>', unsafe_allow_html=True)

    tab_data, tab_con = st.tabs(["Project & Engagement Data", "Project Constraints"])

    with tab_data:
        # ---------- LEVEL 1 — PROJECT ----------
        st.subheader("Project Information")
        proj = st.session_state.project
        pm = proj["metadata"]
        with st.container(border=True):
            with st.form("project_form"):
                c1, c2, c3 = st.columns(3)
                p_name = c1.text_input("Project Name", pm["project_name"])
                p_client = c2.text_input("Client", pm["client"])
                p_phase = c3.text_input("Project Phase", pm["project_phase"])
                p_desc = st.text_area("Project Description (optional)",
                                      pm["project_description"], height=60)
                if st.form_submit_button("Save Project Information", type="primary"):
                    proj["metadata"] = {"project_name": p_name, "client": p_client,
                                        "project_phase": p_phase,
                                        "project_description": p_desc}
                    st.rerun()
            st.markdown(f'<div class="ces-meta">{proj["project_id"]} · Project '
                        'metadata applies to all engagement activities in this '
                        'project.</div>', unsafe_allow_html=True)

        # ---------- LEVEL 2 — ENGAGEMENT ACTIVITIES ----------
        st.subheader("Engagement Activities")
        st.caption("A project can contain multiple engagement activities (mapping "
                   "exercises, surveys, events, focus groups…). One activity can "
                   "contain one or many uploaded files.")
        for activity in all_activities():
            render_activity_card(activity)
        if st.button("＋ Add Engagement Activity"):
            st.session_state.activity_seq += 1
            all_activities().append({
                "activity_id": f"ENG-{st.session_state.activity_seq:03d}",
                "metadata": {k: "" for k in DEFAULT_COMAP_ACTIVITY},
                "dimension": "Other",
                "datasets": [], "combined": None})
            st.rerun()

    # ---------- PROJECT CONSTRAINTS (project level) ----------
    with tab_con:
        st.subheader("Project Constraints")
        st.markdown('<div class="ces-note-human">Constraints belong to the '
                    '<b>project</b>, not to an engagement activity or file. They can '
                    'be linked to themes, evidence, and decisions. A participant '
                    'mentioning Measure LC in a comment is evidence about how that '
                    'participant understands the constraint — it is <b>not</b> the '
                    'authoritative constraint itself.</div>', unsafe_allow_html=True)
        with st.expander("＋ Add Constraint",
                         expanded=(len(st.session_state.constraints) == 0)):
            with st.form("constraint_form", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    cn_name = st.text_input("Constraint Name",
                                            placeholder="e.g. Measure LC")
                    cn_type = st.selectbox("Constraint Type", CONSTRAINT_TYPES)
                    cn_source = st.text_input(
                        "Source", placeholder="e.g. Measure LC (2014) ballot text")
                    cn_status = st.selectbox("Status",
                                             ["Active", "Pending", "Superseded"])
                with c2:
                    cn_desc = st.text_area("Description", height=88)
                    cn_phase = st.text_input("Relevant Phase",
                                             placeholder="e.g. Phase 1–4")
                    cn_notes = st.text_input("Notes")
                if st.form_submit_button("Save Constraint", type="primary"):
                    if cn_name.strip():
                        st.session_state.constraint_seq += 1
                        st.session_state.constraints.append({
                            "id": f"CON-{st.session_state.constraint_seq:03d}",
                            "name": cn_name.strip(), "type": cn_type,
                            "description": cn_desc.strip(),
                            "source": cn_source.strip(),
                            "phase": cn_phase.strip(), "status": cn_status,
                            "notes": cn_notes.strip()})
                        st.rerun()
                    else:
                        st.error("A constraint needs at least a name.")
        if not st.session_state.constraints:
            st.caption("No constraints documented yet.")
        for con in st.session_state.constraints:
            with st.container(border=True):
                st.markdown(f"**{con['id']} — {con['name']}**")
                st.markdown(
                    pills((con["type"], "conflict" if ("Legal" in con["type"] or
                                                       "Voter" in con["type"])
                           else "human"),
                          (con["status"], "validated" if con["status"] == "Active"
                           else "review")), unsafe_allow_html=True)
                if con["description"]:
                    st.write(con["description"])
                st.markdown(f'<div class="ces-meta">Source: {con["source"] or "—"} · '
                            f'Phase: {con["phase"] or "—"}'
                            + (f' · Notes: {con["notes"]}' if con["notes"] else "")
                            + '</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE 02 — ANALYSIS SETUP
# ----------------------------------------------------------------------------

def module_availability(caps):
    """Which playground modules the data can support, with reasons when not."""
    available, unavailable = {}, {}
    for mid, (label, req, reason) in MODULE_DEFS.items():
        if req is None or caps.get(req):
            available[mid] = label
        else:
            unavailable[mid] = (label, reason)
    return available, unavailable


def recommended_modules(caps, goals):
    """Recommended module set from data capabilities + user goals."""
    rec = ["overview"]
    if caps.get("text"):
        rec += ["comments", "themes", "theme_map"]
    if caps.get("multi_group"):
        rec.append("compare")
    if caps.get("coords") and "Spatial patterns" in goals:
        rec.append("map")
    if caps.get("dates") and "Changes over time" in goals:
        rec.append("timeline")
    if caps.get("rankings") and "Priorities" in goals:
        rec.append("rankings")
    if caps.get("demographics") and "Demographic differences" in goals:
        rec.append("stakeholders")
    return rec


def render_analysis_card(an):
    """One saved analysis, with its brief and provenance."""
    activity = get_activity(an["activity_id"])
    dim = an["comparison_dimension"]
    is_active = an["analysis_id"] == st.session_state.active_analysis_id
    with st.container(border=True):
        st.markdown(pills(
            (an["analysis_id"], "gray"),
            ("Active in Playground", "validated") if is_active
            else ("Saved", "human"),
            (dim["name"] if dim["name"] != COMBINED_DIMENSION
             else "Combined datasets", "gray")), unsafe_allow_html=True)
        st.markdown(f"### {an['analysis_name']}")
        st.markdown(
            f'<div class="ces-meta">'
            f'<b>Activity:</b> {activity_name(an["activity_id"])} · '
            f'<b>Datasets:</b> {", ".join(dim["values"]) or ", ".join(an["dataset_ids"])} · '
            f'<b>Unit:</b> {an["unit_of_analysis"]}</div>',
            unsafe_allow_html=True)
        if an["purpose"]:
            st.markdown(f'<div class="ces-note-human"><b>Purpose:</b> '
                        f'{an["purpose"]}</div>', unsafe_allow_html=True)
        if an["questions"]:
            with st.expander(f"Questions to investigate ({len(an['questions'])})"):
                for q in an["questions"]:
                    st.markdown(f"- {q}")
        mods = [MODULE_DEFS[m][0] for m in an["enabled_modules"]
                if m in MODULE_DEFS]
        st.markdown("Enabled modules: " +
                    "".join(pill(m, "human") for m in mods),
                    unsafe_allow_html=True)
        n_themes = len([t for t in st.session_state.themes
                        if t.get("analysis_id") == an["analysis_id"]])
        n_ev = len([e for e in st.session_state.evidence
                    if e.get("analysis_id") == an["analysis_id"]])
        st.markdown(f'<div class="ces-meta">{n_themes} validated themes · '
                    f'{n_ev} evidence items produced by this analysis</div>',
                    unsafe_allow_html=True)
        b1, b2, _ = st.columns([1.2, 1, 2.4])
        if b1.button("Open in Playground", key=f"open-{an['analysis_id']}",
                     type="primary"):
            st.session_state.active_analysis_id = an["analysis_id"]
            st.toast(f'“{an["analysis_name"]}” is now the active analysis — '
                     'open 03 Insights Playground.')
            st.rerun()
        if b2.button("Delete", key=f"del-{an['analysis_id']}",
                     disabled=(n_themes > 0 or n_ev > 0)):
            st.session_state.analyses = [
                a for a in st.session_state.analyses
                if a["analysis_id"] != an["analysis_id"]]
            if is_active:
                st.session_state.active_analysis_id = None
            st.rerun()
        if n_themes or n_ev:
            b2.caption("Has themes/evidence — cannot delete.")


def render_analysis_wizard(draft):
    """AI-assisted Analysis Brief: AI suggests, the human confirms everything."""
    activity = get_activity(draft["activity_id"])
    datasets = [get_dataset(activity, d) for d in draft["dataset_ids"]]
    datasets = [d for d in datasets if d]
    caps = draft["caps"]
    sug = draft["sug"]
    ai_pill = ("AI Suggested", "ai") if sug["source"] == "llm" \
        else ("Suggested (rule-based — no AI key configured)", "review")

    st.markdown("---")
    st.subheader("Analysis Brief")
    st.markdown(pills(ai_pill) + f'<span style="color:{C["text2"]};'
                'font-size:13px;"> Everything below is a suggestion until you '
                'confirm it. You can edit every field.</span>',
                unsafe_allow_html=True)
    st.markdown('<div class="ces-note-ai">' +
                sug["intro"].replace("\n", "<br>") + "</div>",
                unsafe_allow_html=True)
    with st.expander("Data Capabilities (profiled from the selected datasets)"):
        st.markdown(capabilities_html(caps), unsafe_allow_html=True)

    a_name = st.text_input(
        "Analysis Name", draft.get("name_default", ""),
        placeholder="e.g. Housing Across Scenarios", key="wiz-name")

    # ---- Q1: what does each dataset represent? ----
    st.markdown("#### 1 · What does each dataset represent?")
    dim_default = sug["dimension"] if sug["dimension"] in DIMENSION_TYPES else "Other"
    dim = st.selectbox(
        "Dataset Dimension", DIMENSION_TYPES,
        index=DIMENSION_TYPES.index(dim_default), key="wiz-dim",
        help="What distinguishes one dataset from another within this analysis.")
    if dim == dim_default:
        st.markdown(pills(("AI suggests: " + dim_default, "ai")),
                    unsafe_allow_html=True)
    ds_values = {}
    if dim != COMBINED_DIMENSION:
        for d in datasets:
            ds_values[d["dataset_id"]] = st.text_input(
                f'{d["source_file"]} →',
                sug["dataset_values"].get(d["dataset_id"], ""),
                key=f'wiz-dv-{d["dataset_id"]}')
    else:
        st.caption("The selected files will simply be combined — no "
                   "comparison dimension.")

    # ---- Q2: what was this engagement trying to learn? ----
    st.markdown("#### 2 · What was this engagement activity trying to understand?")
    purpose = st.text_area("Analysis Purpose", sug["purpose"], height=80,
                           key="wiz-purpose")

    # ---- Q3: what are you trying to learn now? ----
    st.markdown("#### 3 · What would you like to understand from these datasets?")
    goal_opts = [g for g, req in GOAL_DEFS if req is None or caps.get(req)]
    hidden = [g for g, req in GOAL_DEFS if req is not None and not caps.get(req)]
    goals = st.multiselect(
        "Desired pattern types", goal_opts,
        default=[g for g in sug["goals"] if g in goal_opts], key="wiz-goals")
    if hidden:
        st.caption("Not offered (unsupported by the available data): "
                   + ", ".join(hidden))

    # ---- Q4: primary unit of analysis ----
    st.markdown("#### 4 · Primary unit of analysis")
    unit_default = sug["unit"] if sug["unit"] in UNIT_TYPES else "Other"
    unit = st.selectbox(
        "What should the analysis primarily treat as the unit of input?",
        UNIT_TYPES, index=UNIT_TYPES.index(unit_default), key="wiz-unit")
    st.caption("Multiple rows can share one response ID — the app always "
               "reports both comment counts and unique response-ID counts, "
               "and never assumes one row equals one participant.")

    # ---- Q5: what should be compared? ----
    compare_values = []
    if dim != COMBINED_DIMENSION and len(datasets) > 1:
        st.markdown("#### 5 · What should be compared?")
        vals = [v for v in ds_values.values() if v.strip()]
        compare_values = st.multiselect(
            f"Compare by {dim}", vals, default=vals, key="wiz-compare")
    elif len(datasets) <= 1:
        st.caption("Only one dataset selected — comparison is not applicable.")

    # ---- Q6: questions to investigate ----
    st.markdown("#### 6 · Questions to investigate")
    st.markdown(pills(ai_pill) + f'<span style="color:{C["text2"]};'
                'font-size:13px;"> Edit, delete, or add your own. These stay '
                'visible in the Playground.</span>', unsafe_allow_html=True)
    for q in list(draft["questions"]):
        c1, c2 = st.columns([6, 1])
        q["text"] = c1.text_input(
            "Question", q["text"], key=f'wiz-q-{q["qid"]}',
            label_visibility="collapsed")
        if c2.button("✕", key=f'wiz-qdel-{q["qid"]}'):
            draft["questions"] = [x for x in draft["questions"]
                                  if x["qid"] != q["qid"]]
            st.rerun()
    if st.button("＋ Add Question"):
        draft["qseq"] += 1
        draft["questions"].append({"qid": draft["qseq"], "text": ""})
        st.rerun()

    # ---- Q7: relevant project constraints ----
    st.markdown("#### 7 · Relevant project constraints")
    cons = st.session_state.constraints
    if not cons:
        st.caption("No constraints documented yet — add them in "
                   "01 Data + Context.")
        confirmed_cons = []
    else:
        sug_cons = [c["id"] for c in cons if c["name"] in sug["constraints"]]
        if sug_cons:
            st.markdown(pills(("AI suggests as potentially relevant", "ai")) +
                        " " + ", ".join(
                            f'{cid} — '
                            f'{next(c["name"] for c in cons if c["id"] == cid)}'
                            for cid in sug_cons),
                        unsafe_allow_html=True)
        confirmed_cons = st.multiselect(
            "Constraints you confirm as relevant to this analysis",
            [c["id"] for c in cons], default=sug_cons,
            format_func=lambda cid: f"{cid} — "
            f"{next(c['name'] for c in cons if c['id'] == cid)}",
            key="wiz-cons")
    notes = st.text_area("Analysis Notes (optional)", "", height=60,
                         key="wiz-notes")

    # ---- recommended playground ----
    st.markdown("---")
    st.subheader("Recommended Playground")
    st.caption("Generated from the available data + your analysis goals. "
               "Toggle modules on or off.")
    available, unavailable = module_availability(caps)
    if dim == COMBINED_DIMENSION or len(compare_values) < 2:
        if "compare" in available:
            del available["compare"]
            unavailable["compare"] = (
                "Compare", "Fewer than two comparison values are selected.")
    rec = recommended_modules(caps, goals)
    enabled = []
    mcols = st.columns(min(4, max(1, len(available))))
    for i, (mid, label) in enumerate(available.items()):
        with mcols[i % len(mcols)]:
            if st.checkbox(label, value=(mid in rec), key=f"wiz-mod-{mid}"):
                enabled.append(mid)
    if unavailable:
        st.markdown("**Unavailable**")
        for mid, (label, reason) in unavailable.items():
            st.markdown(f'<div class="ces-meta"><span style="color:#7c4f4e;">'
                        f'✕</span> <b>{label}</b> — {reason}</div>',
                        unsafe_allow_html=True)

    # ---- generate ----
    st.markdown("---")
    g1, g2 = st.columns([1, 1])
    if g1.button("GENERATE PLAYGROUND", type="primary", key="wiz-generate"):
        if not a_name.strip():
            st.error("The analysis needs a name.")
            return
        if dim != COMBINED_DIMENSION and not any(
                v.strip() for v in ds_values.values()):
            st.error(f"Provide at least one {dim} value, or choose "
                     f"“{COMBINED_DIMENSION}”.")
            return
        questions = [q["text"].strip() for q in draft["questions"]
                     if q["text"].strip()]
        if dim == COMBINED_DIMENSION:
            values = ["All records"]
        else:
            values = compare_values or [v for v in ds_values.values() if v.strip()]
        analysis = {
            "analysis_id": next_id("analysis_seq", "AN-"),
            "analysis_name": a_name.strip(),
            "activity_id": draft["activity_id"],
            "dataset_ids": draft["dataset_ids"],
            "comparison_dimension": {"name": dim, "values": values,
                                     "dataset_values": {
                                         k: v.strip() for k, v in
                                         ds_values.items()}},
            "purpose": purpose.strip(),
            "unit_of_analysis": unit,
            "questions": questions,
            "desired_patterns": goals,
            "enabled_modules": enabled or ["overview"],
            "constraint_ids": confirmed_cons,
            "ai_suggested_constraint_ids": [
                c["id"] for c in cons
                if c["name"] in sug["constraints"]] if cons else [],
            "capabilities": caps,
            "notes": notes.strip(),
            "brief_source": sug["source"],
            "created": datetime.date.today().isoformat(),
            "cluster_run_done": False,
            "cluster_scope_desc": "",
        }
        st.session_state.analyses.append(analysis)
        st.session_state.active_analysis_id = analysis["analysis_id"]
        st.session_state.analysis_draft = None
        st.toast(f'Analysis {analysis["analysis_id"]} created — the Insights '
                 'Playground is now configured for it.')
        st.rerun()
    if g2.button("Cancel", key="wiz-cancel"):
        st.session_state.analysis_draft = None
        st.rerun()


def page_analysis_setup():
    st.title("Analysis Setup")
    st.markdown(f'<p style="color:{C["text2"]};margin-top:-8px;">Define what '
                'you want to learn from your engagement data. An Analysis '
                'connects selected datasets to your analytical goals and '
                'configures the Insights Playground. AI helps configure — '
                'you confirm everything.</p>', unsafe_allow_html=True)

    processed = [a for a in all_activities() if a["combined"] is not None]
    if not processed:
        st.markdown('<div class="ces-note-human">No processed engagement data '
                    'yet. Go to <b>01 Data + Context</b>, upload files into an '
                    'engagement activity, and press <b>Process Activity '
                    'Datasets</b>.</div>', unsafe_allow_html=True)
        return

    # ---------- existing analyses ----------
    if st.session_state.analyses:
        st.subheader("Analyses")
        st.caption("The same dataset may be reused in multiple analyses — no "
                   "re-uploading. Each analysis keeps its own playground, "
                   "themes, and evidence.")
        for an in st.session_state.analyses:
            render_analysis_card(an)

    # ---------- new analysis ----------
    st.subheader("＋ New Analysis")
    with st.container(border=True):
        act_ids = [a["activity_id"] for a in processed]
        sel_act = st.selectbox("Engagement Activity", act_ids,
                               format_func=activity_name, key="new-an-act")
        activity = get_activity(sel_act)
        ds_opts = {d["dataset_id"]: (d["dataset_name"] or d["source_file"])
                   for d in activity["datasets"]
                   if d["include"] and d["df"] is not None}
        sel_ds = st.multiselect("Datasets to analyze", list(ds_opts),
                                default=list(ds_opts),
                                format_func=lambda k: ds_opts[k],
                                key="new-an-ds")
        provider, _ = llm_provider()
        st.caption("AI will inspect the engagement metadata, dataset "
                   "metadata, field structure, and data capabilities, then "
                   "suggest a draft Analysis Brief for you to confirm."
                   if provider else
                   "No AI key configured — rule-based suggestions will be "
                   "used instead (everything remains editable).")
        if st.button("Begin Analysis Setup", type="primary",
                     disabled=(len(sel_ds) == 0), key="new-an-begin"):
            datasets = [get_dataset(activity, d) for d in sel_ds]
            caps = data_capabilities(datasets)
            with st.spinner("Inspecting datasets and drafting the Analysis "
                            "Brief…"):
                sug = ai_analysis_suggestions(activity, datasets, caps)
            topic = next((d["topic"] for d in datasets if d["topic"]), "")
            dim_word = sug["dimension"].split(" /")[0]
            name_default = (f"{topic} Across {dim_word}s".strip()
                            if topic and len(datasets) > 1 else
                            (activity["metadata"]["activity_name"]
                             or "New Analysis"))
            st.session_state.analysis_draft = {
                "activity_id": sel_act, "dataset_ids": sel_ds,
                "caps": caps, "sug": sug, "name_default": name_default,
                "questions": [{"qid": i, "text": q}
                              for i, q in enumerate(sug["questions"])],
                "qseq": len(sug["questions"]),
            }
            st.rerun()

    draft = st.session_state.analysis_draft
    if draft:
        render_analysis_wizard(draft)


# ----------------------------------------------------------------------------
# PAGE 03 — INSIGHTS PLAYGROUND (dynamically configured by the active ANALYSIS)
# ----------------------------------------------------------------------------

def reaction_chart(df, group_col="dim_value", group_label="Group"):
    groups = sorted(df[group_col].unique())
    fig = go.Figure()
    for reaction in ("approve", "disapprove", "none"):
        vals = [int(((df[group_col] == g) & (df["reaction"] == reaction)).sum())
                for g in groups]
        fig.add_bar(name=reaction.title(), x=groups, y=vals,
                    marker_color=REACTION_COLOR[reaction],
                    text=vals, textposition="outside")
    fig.update_layout(
        barmode="group", height=360, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color=C["text"], size=13),
        legend=dict(orientation="h", y=1.12, title="Participant Reaction"),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        xaxis=dict(linecolor=C["border"], title=group_label))
    return fig


def group_pct_table(df, dim_label="Group"):
    rows = []
    for s in sorted(df["dim_value"].unique()):
        sdf = df[df["dim_value"] == s]
        cts = reaction_counts(sdf)
        n = max(1, len(sdf))
        rows.append({dim_label: s, "Comments": len(sdf),
                     "Unique response IDs": sdf["response_id"].nunique(),
                     "Approve": f'{cts["approve"]} ({cts["approve"]/n:.0%})',
                     "Disapprove": f'{cts["disapprove"]} ({cts["disapprove"]/n:.0%})',
                     "None": f'{cts["none"]} ({cts["none"]/n:.0%})'})
    return pd.DataFrame(rows)


def constraint_auto_suggestions(cluster):
    texts = " ".join(records_for(cluster["record_ids"])["comment"].str.lower())
    hits = []
    for con in st.session_state.constraints:
        if con["id"] in cluster["constraint_links"]:
            continue
        name = con["name"].strip().lower()
        if name and name in texts:
            hits.append(con)
    return hits


def render_constraint_links(cluster, key_prefix):
    for con in constraint_auto_suggestions(cluster):
        cluster["constraint_links"][con["id"]] = "ai"
    links = {cid: s for cid, s in cluster["constraint_links"].items()
             if s != "dismissed"}
    if not links:
        return
    st.markdown("**Potentially relevant constraints**")
    for cid, state in links.items():
        con = next((c for c in st.session_state.constraints if c["id"] == cid), None)
        if not con:
            continue
        with st.container(border=True):
            if state == "ai":
                st.markdown(pills(("AI Suggested Relationship", "ai")) +
                            f" **{con['id']} — {con['name']}**",
                            unsafe_allow_html=True)
                st.caption("Participants in this cluster mention this project "
                           "constraint by name. AI only surfaces it — a human "
                           "decides whether it is relevant.")
                b1, b2, b3 = st.columns(3)
                if b1.button("Review Constraint", key=f"{key_prefix}-rc-{cid}"):
                    st.info(f"{con['id']} · {con['type']} · {con['status']}  \n"
                            f"{con['description'] or '(no description)'}  \n"
                            f"Source: {con['source'] or '—'}")
                if b2.button("Mark as Relevant", key=f"{key_prefix}-mr-{cid}"):
                    cluster["constraint_links"][cid] = "confirmed"
                    st.rerun()
                if b3.button("Dismiss", key=f"{key_prefix}-dm-{cid}"):
                    cluster["constraint_links"][cid] = "dismissed"
                    st.rerun()
            else:
                st.markdown(pills(("Human Confirmed", "validated")) +
                            f" **{con['id']} — {con['name']}**",
                            unsafe_allow_html=True)


def render_cluster_card(cluster):
    key = cluster["key"]
    ai = cluster["ai"]
    cts = cluster["counts"]
    with st.container(border=True):
        src_pill = ("AI Suggested Theme", "ai") if ai["source"] == "llm" \
            else ("Keyword Cluster — AI interpretation unavailable", "review")
        status_pill = []
        if cluster["status"] == "validated":
            status_pill = [("Human Validated", "validated")]
        st.markdown(pills(src_pill, (cluster["group"], "gray"), *status_pill),
                    unsafe_allow_html=True)
        st.markdown(f"### {ai['name']}")
        st.markdown(
            f'<div class="ces-meta">{cluster["n_comments"]} related comments · '
            f'{cluster["n_respondents"]} unique response IDs · '
            f'Datasets: {", ".join(cluster["dataset_ids"])}<br>'
            f'Reaction — Approve {cts["approve"]} · Disapprove {cts["disapprove"]} '
            f'· None {cts["none"]}</div>', unsafe_allow_html=True)
        note_cls = "ces-note-ai" if ai["source"] == "llm" else "ces-note-warn"
        label = "AI Summary" if ai["source"] == "llm" else "No AI interpretation"
        st.markdown(f'<div class="{note_cls}"><b>{label}:</b> {ai["summary"]}</div>',
                    unsafe_allow_html=True)
        st.markdown("Suggested tags: " +
                    "".join(pill(t, "ai") for t in ai.get("tags", [])),
                    unsafe_allow_html=True)
        st.caption("Title, summary, and tags are AI-generated suggestions. Counts "
                   "are calculated in Python from the associated record IDs.")
        render_constraint_links(cluster, key)

        if cluster["status"] == "ai-suggested":
            b = st.columns(5)
            if b[0].button("View Comments", key=f"vc-{key}"):
                st.session_state.viewing_cluster[key] = \
                    not st.session_state.viewing_cluster.get(key, False)
            with b[1].popover("Edit Theme"):
                new_name = st.text_input("Working name", ai["name"],
                                         key=f"edit-name-{key}")
                if st.button("Apply", key=f"edit-apply-{key}"):
                    ai.setdefault("original_name", ai["name"])
                    ai["name"] = new_name
                    st.rerun()
            with b[2].popover("Merge"):
                others = [k for k, c in st.session_state.clusters.items()
                          if k != key and c["status"] == "ai-suggested"
                          and c.get("analysis_id") == cluster.get("analysis_id")]
                if others:
                    target = st.selectbox(
                        "Merge into", others,
                        format_func=lambda k:
                        f'{st.session_state.clusters[k]["ai"]["name"]} '
                        f'({st.session_state.clusters[k]["group"]})',
                        key=f"merge-sel-{key}")
                    if st.button("Merge clusters", key=f"merge-do-{key}"):
                        tgt = st.session_state.clusters[target]
                        tgt["record_ids"] = sorted(set(tgt["record_ids"])
                                                   | set(cluster["record_ids"]))
                        sub = records_for(tgt["record_ids"])
                        tgt["n_comments"] = len(sub)
                        tgt["n_respondents"] = sub["response_id"].nunique()
                        tgt["counts"] = reaction_counts(sub)
                        tgt["dataset_ids"] = sorted(sub["dataset_id"].unique())
                        tgt["source_files"] = sorted(sub["source_file"].unique())
                        maj = "approve" if tgt["counts"]["approve"] >= \
                            tgt["counts"]["disapprove"] else "disapprove"
                        minr = "disapprove" if maj == "approve" else "approve"
                        tgt["majority"] = maj
                        tgt["counter_ids"] = sub[sub["reaction"] == minr][
                            "record_id"].tolist()
                        tgt.setdefault("merged_from", []).append(ai["name"])
                        cluster["status"] = "merged"
                        st.rerun()
                else:
                    st.caption("No other AI clusters to merge with.")
            if b[3].button("Reject", key=f"rej-{key}"):
                cluster["status"] = "rejected"
                st.rerun()
            if b[4].button("Validate", key=f"val-{key}", type="primary"):
                st.session_state.validating_cluster = key
                st.rerun()

        if st.session_state.viewing_cluster.get(key):
            sub = records_for(cluster["record_ids"])
            sort = st.selectbox("Sort by", ["Approve first", "Disapprove first",
                                            "None first", "Response ID"],
                                key=f"sort-{key}")
            if sort == "Response ID":
                sub = sub.sort_values("response_id")
            else:
                first = sort.split()[0].lower()
                sub = sub.sort_values("reaction",
                                      key=lambda s: (s != first).astype(int))
            st.markdown("**Representative comments** (closest to cluster center)")
            for rid in cluster["rep_ids"][:3]:
                r = get_record(rid)
                if r is not None:
                    comment_card(r, f"rep-{key}")
            if cluster["counter_ids"]:
                st.markdown(f'<div class="ces-note-warn"><b>Potential '
                            f'counter-evidence ({len(cluster["counter_ids"])}):</b> '
                            f'comments whose reaction opposes the cluster majority '
                            f'({cluster["majority"]}). A common theme is not '
                            'automatically consensus.</div>', unsafe_allow_html=True)
                for rid in cluster["counter_ids"][:5]:
                    r = get_record(rid)
                    if r is not None:
                        comment_card(r, f"ctr-{key}")
            with st.expander(f"All {len(sub)} comments in this cluster"):
                for _, r in sub.iterrows():
                    comment_card(r, f"all-{key}")


def render_validation_form(cluster):
    key = cluster["key"]
    ai = cluster["ai"]
    st.markdown("---")
    st.subheader(f"Validate Theme — {ai['name']}")
    st.markdown('<div class="ces-note-human">Keep the theme name SHORT (2–4 words). '
                'The AI original name and summary are preserved unchanged alongside '
                'your interpretation.</div>', unsafe_allow_html=True)
    val_an = get_analysis(cluster.get("analysis_id"))
    if val_an:
        st.markdown("**Visual Cluster Review** — inspect the grouping before "
                    "validating: comments at the center vs the edge (faded = "
                    "ambiguous), counter-evidence (red ring), reaction "
                    "distribution, and overlap with other themes.")
        if not theme_mini_map(val_an, cluster["record_ids"],
                              cluster["counter_ids"],
                              key=f"valmap-{key}"):
            st.caption("Semantic map unavailable for this analysis "
                       "(too few comments).")
    sub = records_for(cluster["record_ids"])
    label_map = {rid: f'{rid} · {sub[sub["record_id"] == rid]["comment"].iloc[0][:80]}'
                 for rid in cluster["record_ids"]}
    with st.form(f"validate-{key}"):
        name = st.text_input("Theme Name (short)", ai["name"])
        interp = st.text_area("Human Interpretation",
                              placeholder="What does this pattern mean, in your "
                                          "own words?")
        include = st.multiselect("Comments to Include", cluster["record_ids"],
                                 default=cluster["record_ids"],
                                 format_func=lambda r: label_map[r])
        tags = st.multiselect("Tags", sorted(set(all_known_tags())
                                             | set(ai.get("tags", []))),
                              default=ai.get("tags", []))
        new_tag = st.text_input("Add a new tag (optional)")
        notes = st.text_area("Optional Notes", height=60)
        rel_cons = st.multiselect(
            "Relevant Constraints",
            [c["id"] for c in st.session_state.constraints],
            default=[cid for cid, s in cluster["constraint_links"].items()
                     if s == "confirmed"],
            format_func=lambda cid: f"{cid} — "
            f"{next(c['name'] for c in st.session_state.constraints if c['id'] == cid)}")
        c1, c2 = st.columns(2)
        submitted = c1.form_submit_button("Validate Theme", type="primary")
        cancelled = c2.form_submit_button("Cancel")
    if cancelled:
        st.session_state.validating_cluster = None
        st.session_state.validating_in_map = False
        st.rerun()
    if submitted:
        if not interp.strip():
            st.error("A human interpretation is required — that is the point "
                     "of review.")
            return
        if new_tag.strip():
            tags = tags + [new_tag.strip()]
        excluded = [r for r in cluster["record_ids"] if r not in include]
        prov = provenance_from_records(include)
        inc_sub = records_for(include)
        theme_id = next_id("theme_seq", "TH-")
        an = get_analysis(cluster.get("analysis_id"))
        theme = {
            "theme_id": theme_id, "origin": "ai-validated",
            "analysis_id": cluster.get("analysis_id"),
            "dimension": (an["comparison_dimension"]["name"] if an else None),
            "ai_original_name": ai.get("original_name", ai["name"]),
            "ai_original_summary": ai["summary"], "ai_source": ai["source"],
            "name": name.strip(), "interpretation": interp.strip(),
            "record_ids": include, "excluded_record_ids": excluded,
            "counts": reaction_counts(inc_sub),
            "n_respondents": int(inc_sub["response_id"].nunique()),
            "tags": tags, "notes": notes.strip(),
            "counter_ids": [r for r in cluster["counter_ids"] if r in include],
            "constraints": rel_cons,
            "validated": datetime.date.today().isoformat(),
            "status": "HUMAN VALIDATED", "cluster_key": key, **prov}
        st.session_state.themes.append(theme)
        for rid in include:
            for t in tags:
                add_tag(rid, t, "ai" if t in ai.get("tags", []) else "human")
        cluster["status"] = "validated"
        cluster["theme_id"] = theme_id
        st.session_state.validating_cluster = None
        st.session_state.validating_in_map = False
        st.toast(f"Theme {theme_id} validated — see the Evidence Library")
        st.rerun()


def render_create_human_theme(df):
    with st.expander("＋ Create Theme (human-created, no AI)"):
        label_map = {rid: f'{rid} · {c[:80]}' for rid, c in
                     zip(df["record_id"], df["comment"])}
        with st.form("human-theme"):
            name = st.text_input("Theme Name (short)")
            interp = st.text_area("Interpretation")
            include = st.multiselect("Select Comments", df["record_id"].tolist(),
                                     format_func=lambda r: label_map[r])
            tags = st.multiselect("Tags", all_known_tags())
            new_tag = st.text_input("Add a new tag (optional)")
            rel_cons = st.multiselect(
                "Relevant Constraints",
                [c["id"] for c in st.session_state.constraints],
                format_func=lambda cid: f"{cid} — "
                f"{next(c['name'] for c in st.session_state.constraints if c['id'] == cid)}")
            if st.form_submit_button("Save Theme", type="primary"):
                if not name.strip() or not include:
                    st.error("A human theme needs a name and at least one comment.")
                else:
                    if new_tag.strip():
                        tags = tags + [new_tag.strip()]
                    prov = provenance_from_records(include)
                    inc_sub = records_for(include)
                    theme_id = next_id("theme_seq", "TH-")
                    an = active_analysis()
                    st.session_state.themes.append({
                        "theme_id": theme_id, "origin": "human",
                        "analysis_id": an["analysis_id"] if an else None,
                        "dimension": (an["comparison_dimension"]["name"]
                                      if an else None),
                        "ai_original_name": None, "ai_original_summary": None,
                        "ai_source": None, "name": name.strip(),
                        "interpretation": interp.strip(),
                        "record_ids": include, "excluded_record_ids": [],
                        "counts": reaction_counts(inc_sub),
                        "n_respondents": int(inc_sub["response_id"].nunique()),
                        "tags": tags, "notes": "", "counter_ids": [],
                        "constraints": rel_cons,
                        "validated": datetime.date.today().isoformat(),
                        "status": "HUMAN VALIDATED", "cluster_key": None, **prov})
                    for rid in include:
                        for t in tags:
                            add_tag(rid, t, "human")
                    st.toast(f"Human theme {theme_id} saved")
                    st.rerun()


def cross_group_patterns(analysis):
    """AI-computed pattern proposals from keyword overlap between one
    analysis's clusters, across its comparison-dimension values."""
    dim_word = analysis["comparison_dimension"]["name"].split(" /")[0]
    clusters = list(analysis_clusters(analysis["analysis_id"]).values())
    by_grp = {}
    for c in clusters:
        by_grp.setdefault(c["group"], []).append(c)
    grps = sorted(by_grp)
    patterns, seen = [], set()
    for i, s1 in enumerate(grps):
        for c1 in by_grp[s1]:
            group = [c1]
            kws1 = set(c1["keywords"])
            for s2 in grps[i + 1:]:
                best, best_j = None, 0.0
                for c2 in by_grp[s2]:
                    inter = len(kws1 & set(c2["keywords"]))
                    union = len(kws1 | set(c2["keywords"]))
                    j = inter / union if union else 0
                    if j > best_j:
                        best, best_j = c2, j
                if best is not None and best_j >= 0.15:
                    group.append(best)
            gkey = tuple(sorted(c["key"] for c in group))
            if len(group) >= 2 and gkey not in seen:
                seen.add(gkey)
                n_grp = len({c["group"] for c in group})
                if n_grp == len(grps) and len(grps) >= 3:
                    rel = f"Appears Across All {dim_word}s"
                else:
                    dis = {c["group"]: (c["counts"]["disapprove"] /
                                        max(1, c["n_comments"])) for c in group}
                    strongest = max(dis, key=dis.get)
                    rel = (f"Stronger in {strongest}"
                           if max(dis.values()) - min(dis.values()) > 0.2
                           else f"Shared Across {dim_word}s")
                patterns.append({"key": "|".join(gkey), "relationship": rel,
                                 "clusters": group,
                                 "shared_keywords": sorted(set.intersection(
                                     *[set(c["keywords"]) for c in group]))})
    for c in clusters:
        if not any(c in p["clusters"] for p in patterns):
            patterns.append({"key": c["key"],
                             "relationship": f"Mostly {dim_word}-Specific",
                             "clusters": [c],
                             "shared_keywords": c["keywords"][:4]})
    return patterns


# ----------------------------------------------------------------------------
# THEME MAP (semantic map of comments — AI suggests, humans inspect & correct)
# ----------------------------------------------------------------------------

# Theme/cluster styling per status. Reaction color on points is authoritative
# and never replaced — these colors are used for boundaries/labels only.
GROUP_STATUS_COLOR = {"ai": C["purple"], "human": C["blue"],
                      "validated": C["green"]}
GROUP_STATUS_LABEL = {"ai": "AI Suggested Theme",
                      "human": "Human Created Theme",
                      "validated": "Human Validated Theme"}
AMBIGUITY_THRESHOLD = 0.4  # normalized similarity below this = ambiguous

SEMANTIC_AXES_NOTE = (
    "Position represents similarity in comment meaning. Nearby comments use "
    "semantically similar language. The axes themselves do not represent "
    "predefined planning concepts.")


def get_semantic_coords(analysis, adf):
    """2D semantic coordinates for every comment in one analysis:
    TF-IDF → TruncatedSVD (LSA). Cached per analysis; recomputed when the
    record set changes. Axes are unnamed semantic dimensions — no conceptual
    meaning is invented for them."""
    from sklearn.decomposition import TruncatedSVD
    cache = st.session_state.setdefault("semantic_maps", {})
    sig = (analysis["analysis_id"], len(adf),
           adf["record_id"].iloc[0], adf["record_id"].iloc[-1])
    entry = cache.get(analysis["analysis_id"])
    if entry and entry["sig"] == sig:
        return entry
    texts = adf["comment"].fillna("").astype(str).tolist()
    if len(texts) < 4:
        return None
    vec = TfidfVectorizer(stop_words="english", max_features=2500,
                          ngram_range=(1, 2), min_df=1)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return None
    if X.shape[1] < 3:
        return None
    coords = TruncatedSVD(n_components=2, random_state=42).fit_transform(X)
    entry = {"sig": sig,
             "x": dict(zip(adf["record_id"], coords[:, 0].round(4))),
             "y": dict(zip(adf["record_id"], coords[:, 1].round(4)))}
    cache[analysis["analysis_id"]] = entry
    return entry


def theme_groups(analysis_id):
    """Unified view of this analysis's groupings: AI-suggested clusters plus
    validated/human themes, each with status, records, and counter-evidence."""
    groups = []
    for key, cl in analysis_clusters(analysis_id,
                                     statuses=("ai-suggested",)).items():
        groups.append({"gid": key, "kind": "cluster", "name": cl["ai"]["name"],
                       "status": "ai", "record_ids": cl["record_ids"],
                       "counter_ids": cl["counter_ids"],
                       "confidence": cl.get("confidence", {}), "obj": cl})
    for th in st.session_state.themes:
        if th.get("analysis_id") != analysis_id:
            continue
        status = "validated" if th.get("origin") == "ai-validated" else "human"
        groups.append({"gid": th["theme_id"], "kind": "theme",
                       "name": th["name"], "status": status,
                       "record_ids": th["record_ids"],
                       "counter_ids": th.get("counter_ids", []),
                       "confidence": {}, "obj": th})
    return groups


def refresh_cluster_records(cl, record_ids):
    """Recompute a cluster's calculated fields after a human edits its
    membership. Deterministic Python only; AI name/summary untouched."""
    sub = records_for(sorted(set(record_ids)))
    cl["record_ids"] = sub["record_id"].tolist()
    cl["n_comments"] = int(len(sub))
    cl["n_respondents"] = int(sub["response_id"].nunique()) if len(sub) else 0
    cl["counts"] = reaction_counts(sub) if len(sub) else \
        {"approve": 0, "disapprove": 0, "none": 0}
    maj = "approve" if cl["counts"]["approve"] >= cl["counts"]["disapprove"] \
        else "disapprove"
    minr = "disapprove" if maj == "approve" else "approve"
    cl["majority"] = maj
    cl["counter_ids"] = sub[sub["reaction"] == minr]["record_id"].tolist()
    cl["dataset_ids"] = sorted(sub["dataset_id"].unique().tolist())
    cl["source_files"] = sorted(sub["source_file"].unique().tolist())
    cl["activity_ids"] = sorted(sub["activity_id"].unique().tolist())
    cl["rep_ids"] = [r for r in cl.get("rep_ids", [])
                     if r in cl["record_ids"]]
    cl.setdefault("human_edited", True)


def refresh_theme_records(th, record_ids):
    """Recompute a validated theme's calculated fields after a human edits
    its membership. Provenance is refreshed from the actual records."""
    ids = sorted(set(record_ids))
    sub = records_for(ids)
    th["record_ids"] = ids
    th["counts"] = reaction_counts(sub) if len(sub) else \
        {"approve": 0, "disapprove": 0, "none": 0}
    th["n_respondents"] = int(sub["response_id"].nunique()) if len(sub) else 0
    th["counter_ids"] = [r for r in th.get("counter_ids", []) if r in ids]
    th.update(provenance_from_records(ids))


def add_records_to_group(group, record_ids):
    merged = set(group["record_ids"]) | set(record_ids)
    obj = group["obj"]
    obj["removed_record_ids"] = [r for r in obj.get("removed_record_ids", [])
                                 if r not in set(record_ids)]
    if group["kind"] == "cluster":
        refresh_cluster_records(obj, merged)
    else:
        refresh_theme_records(obj, merged)


def remove_records_from_group(group, record_ids):
    remaining = set(group["record_ids"]) - set(record_ids)
    obj = group["obj"]
    removed = set(obj.get("removed_record_ids", [])) | \
        (set(record_ids) & set(group["record_ids"]))
    obj["removed_record_ids"] = sorted(removed)
    if group["kind"] == "cluster":
        refresh_cluster_records(obj, remaining)
    else:
        refresh_theme_records(obj, remaining)


def _build_theme_map_points(an, view, coords):
    """Plot-ready dataframe: one row per original comment in the current
    view, with semantic coordinates, group membership, confidence, and
    counter-evidence flags. Points never detach from their source records."""
    pts = view.copy()
    pts["semantic_x"] = pts["record_id"].map(coords["x"])
    pts["semantic_y"] = pts["record_id"].map(coords["y"])
    pts = pts.dropna(subset=["semantic_x", "semantic_y"])
    memb, conf, counter = {}, {}, set()
    for g in theme_groups(an["analysis_id"]):
        for rid in g["record_ids"]:
            memb.setdefault(rid, g["gid"])
        conf.update(g.get("confidence", {}))
        counter.update(g["counter_ids"])
    pts["group_id"] = pts["record_id"].map(memb)
    pts["confidence"] = pts["record_id"].map(conf).fillna(1.0)
    pts["is_counter"] = pts["record_id"].isin(counter)
    pts["is_ambiguous"] = pts["confidence"] < AMBIGUITY_THRESHOLD
    return pts


def _selectable_chart(fig, chart_key):
    """Render a plotly chart with click/box/lasso selection; returns the
    selected customdata values (empty when unsupported or nothing selected)."""
    try:
        event = st.plotly_chart(fig, width="stretch", key=chart_key,
                                on_select="rerun",
                                selection_mode=("points", "box", "lasso"))
    except TypeError:
        # older Streamlit without selection events — chart still renders
        st.plotly_chart(fig, width="stretch", key=f"{chart_key}-static")
        return []
    out = []
    try:
        for p in (event.selection.get("points", []) if event else []):
            cd = p.get("customdata")
            if cd is not None:
                out.append(str(cd))
    except Exception:
        pass
    return sorted(set(out))


def dominant_reaction(counts):
    """Classify a theme's participant reactions (authoritative field, never
    AI sentiment): Mostly Approve / Mostly Disapprove / Mixed.
    Returns (label, fill color, dominant reaction word, dominant share)."""
    total = max(1, sum(counts.values()))
    shares = {r: counts.get(r, 0) / total
              for r in ("approve", "disapprove", "none")}
    if shares["approve"] >= 0.6:
        return "Mostly Approve", C["green"], "Approve", shares["approve"]
    if shares["disapprove"] >= 0.6:
        return "Mostly Disapprove", C["red"], "Disapprove", shares["disapprove"]
    top = max(shares, key=shares.get)
    return "Mixed / Divided", C["yellow"], top.title(), shares[top]


# Theme Detail fill colors: how strongly a comment belongs to the theme.
REL_COLOR = {"Core": C["blue"], "Related": C["purple"], "Edge": C["yellow"]}


def theme_local_space(group):
    """The selected theme becomes its own analytical space: TF-IDF + 2D
    reduction recalculated over ONLY this theme's comments, plus cosine
    similarity to the theme centroid. Relationship tertiles (Core / Related /
    Edge) are analytical conveniences, not objective thresholds."""
    from sklearn.decomposition import TruncatedSVD
    ids = sorted(set(group["record_ids"]))
    if not ids:
        return None
    cache = st.session_state.setdefault("theme_spaces", {})
    sig = (len(ids), ids[0], ids[-1], tuple(ids[::max(1, len(ids) // 20)]))
    entry = cache.get(group["gid"])
    if entry and entry["sig"] == sig:
        return entry
    sub = records_for(ids)
    if sub.empty:
        return None
    texts = sub["comment"].fillna("").astype(str).tolist()
    vec = TfidfVectorizer(stop_words="english", max_features=2000,
                          ngram_range=(1, 2), min_df=1)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return None
    n = X.shape[0]
    if n >= 4 and X.shape[1] >= 3:
        coords = TruncatedSVD(n_components=2,
                              random_state=42).fit_transform(X)
    else:
        rng = np.random.RandomState(42)
        coords = rng.uniform(-1, 1, size=(n, 2))
    # cosine similarity to the theme centroid — the measurable score behind
    # Core / Related / Edge and shown per comment on inspection
    c = np.asarray(X.mean(axis=0)).ravel()
    cn = float(np.linalg.norm(c)) or 1.0
    rn = np.sqrt(np.asarray(X.multiply(X).sum(axis=1))).ravel()
    rn[rn == 0] = 1.0
    sims = np.asarray(X.dot(c)).ravel() / (rn * cn)
    if n >= 3:
        q1, q2 = np.quantile(sims, [1 / 3, 2 / 3])
    else:
        q1 = q2 = float(sims.min()) - 1.0  # tiny themes: everything Core
    rel = np.where(sims >= q2, "Core",
                   np.where(sims >= q1, "Related", "Edge"))
    # semantic outliers: unusually far from the theme's own 2D center
    d = np.sqrt(((coords - coords.mean(axis=0)) ** 2).sum(axis=1))
    if n >= 5 and float(d.std()) > 0:
        sem_out = d > d.mean() + 1.5 * d.std()
    else:
        sem_out = np.zeros(n, dtype=bool)
    # top terms at each axis extreme — keyword fallback for axis poles
    terms = np.array(vec.get_feature_names_out())
    kend = max(2, n // 5)

    def _pole_terms(axis, side):
        order = np.argsort(coords[:, axis])
        idx = order[:kend] if side == "neg" else order[-kend:]
        m = np.asarray(X[idx].mean(axis=0)).ravel()
        return terms[np.argsort(m)[::-1][:3]].tolist()

    poles = {ax: {"neg": _pole_terms(ax, "neg"),
                  "pos": _pole_terms(ax, "pos")} for ax in (0, 1)}
    df = pd.DataFrame({"record_id": sub["record_id"].tolist(),
                       "x": coords[:, 0].round(4),
                       "y": coords[:, 1].round(4),
                       "sim": np.round(sims, 3).astype(float),
                       "rel": rel, "sem_outlier": sem_out})
    entry = {"sig": sig, "df": df, "poles": poles}
    cache[group["gid"]] = entry
    return entry


def suggest_axis_interpretation(group, space, axis):
    """AI-SUGGESTED conceptual poles for one semantic axis of a theme's
    comment landscape. LLM when configured, keyword poles otherwise. Always a
    suggestion — the human accepts, renames, or rejects; the map is never
    silently assigned conceptual meanings."""
    sub = records_for(group["record_ids"])[["record_id", "comment"]]
    dfc = space["df"].merge(sub, on="record_id")
    col = "x" if axis == 0 else "y"
    order = dfc.sort_values(col)
    lo = order["comment"].head(5).tolist()
    hi = order["comment"].tail(5).tolist()
    prompt = (
        "Public-engagement comments in one theme have been placed along a "
        "semantic dimension. Below are comments from the two OPPOSITE ends. "
        "Suggest a short conceptual label for each end IF a defensible "
        "interpretation exists.\n\n"
        "LOW end comments:\n" + "\n".join(f"- {t[:200]}" for t in lo) +
        "\n\nHIGH end comments:\n" + "\n".join(f"- {t[:200]}" for t in hi) +
        '\n\nRespond with ONLY JSON: {"neg": "<label for LOW end, 2-6 words>", '
        '"pos": "<label for HIGH end, 2-6 words>", '
        '"defensible": true|false}\n'
        "If the ends do not differ in an interpretable way, set defensible "
        "to false.")
    data = llm_json(prompt, max_tokens=250)
    if data and data.get("defensible") and data.get("neg") and data.get("pos"):
        return {"neg": str(data["neg"]).strip(), "pos": str(data["pos"]).strip(),
                "source": "llm"}
    kw = space["poles"][axis]
    return {"neg": " / ".join(kw["neg"][:2]).title(),
            "pos": " / ".join(kw["pos"][:2]).title(),
            "source": "keywords"}


def suggest_subthemes(group, an):
    """AI-SUGGESTED subthemes inside one theme: KMeans on the theme's own
    text vectors, names from the LLM (clearly-labeled keyword fallback
    otherwise). Suggestions only — the human accepts, renames, merges, or
    rejects."""
    ids = sorted(set(group["record_ids"]))
    sub = records_for(ids)
    if len(sub) < 6:
        return []
    texts = sub["comment"].fillna("").astype(str).tolist()
    vec = TfidfVectorizer(stop_words="english", max_features=2000,
                          ngram_range=(1, 2), min_df=1)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return []
    k = int(min(4, max(2, len(sub) // 12)))
    if X.shape[0] <= k:
        return []
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    terms = np.array(vec.get_feature_names_out())
    provider, _ = llm_provider()
    out = []
    for ci in range(k):
        idx = np.where(labels == ci)[0]
        if len(idx) < 2:
            continue
        ssub = sub.iloc[idx]
        center = np.asarray(km.cluster_centers_[ci]).ravel()
        kws = terms[np.argsort(center)[::-1][:6]].tolist()
        ai = llm_interpret_cluster(ssub["comment"].head(8).tolist(), kws,
                                   f'subthemes of "{group["name"]}"') \
            if provider else None
        name = ai["name"] if ai else " / ".join(w.title() for w in kws[:2])
        out.append({"sid": f'{group["gid"]}-S{ci + 1}', "name": name,
                    "record_ids": ssub["record_id"].tolist(),
                    "keywords": kws, "status": "ai",
                    "source": "llm" if ai else "keywords"})
    return out


def _reaction_spectrum(pts, groups, dim_label):
    """Secondary view: participant reaction (authoritative, never AI
    sentiment) on the horizontal axis, one row per theme — reveals which
    themes lean approve, disapprove, or split internally."""
    pos = {"disapprove": -1.0, "none": 0.0, "approve": 1.0}
    rows = [(g["gid"], g["name"], g["status"], set(g["record_ids"]))
            for g in groups]
    assigned = set().union(*[r[3] for r in rows]) if rows else set()
    un = set(pts["record_id"]) - assigned
    if un:
        rows.append(("__none__", "Not in any theme", "human", un))
    rng = np.random.RandomState(42)
    fig = go.Figure()
    ticks, tickvals = [], []
    plotted = set()
    for i, (gid, name, status, ids) in enumerate(rows):
        gp = pts[pts["record_id"].isin(list(ids - plotted))
                 if gid == "__none__" else pts["record_id"].isin(list(ids))]
        if gp.empty:
            continue
        jitter_x = rng.uniform(-0.32, 0.32, len(gp))
        jitter_y = rng.uniform(-0.28, 0.28, len(gp))
        fig.add_scatter(
            x=gp["reaction"].map(pos).to_numpy() + jitter_x,
            y=np.full(len(gp), i) + jitter_y,
            mode="markers", showlegend=False,
            customdata=gp["record_id"],
            marker=dict(color=gp["reaction"].map(REACTION_COLOR), size=8,
                        opacity=0.85, line=dict(color="#FFFFFF", width=0.5)),
            hovertemplate=("%{customdata} · " + name[:30]
                           + "<br>“%{text}”<extra></extra>"),
            text=gp["comment"].str.slice(0, 90))
        ticks.append(f"{name[:34]}")
        tickvals.append(i)
        if gid != "__none__":
            plotted |= ids
    fig.update_layout(
        height=max(320, 90 * max(1, len(tickvals))),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color=C["text"], size=12),
        xaxis=dict(tickmode="array", tickvals=[-1, 0, 1],
                   ticktext=["Disapprove", "None", "Approve"],
                   range=[-1.6, 1.6], gridcolor=C["border"], zeroline=False,
                   title="Participant Reaction (participant-provided — "
                         "not AI sentiment)"),
        yaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticks,
                   gridcolor=C["border"], zeroline=False, autorange="reversed"))
    st.plotly_chart(fig, width="stretch", key="tmap-spectrum")


def theme_mini_map(an, record_ids, counter_ids, key):
    """Compact visual cluster review used inside theme validation: the
    theme's comments highlighted in semantic space, counter-evidence ringed,
    everything else faded."""
    adf = analysis_df(an)
    if adf is None or adf.empty:
        return False
    coords = get_semantic_coords(an, adf)
    if not coords:
        return False
    pts = _build_theme_map_points(an, adf, coords)
    focus = set(record_ids)
    fig = go.Figure()
    other = pts[~pts["record_id"].isin(list(focus))]
    if len(other):
        fig.add_scatter(x=other["semantic_x"], y=other["semantic_y"],
                        mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(color="#D6D6D2", size=6, opacity=0.4))
    mine = pts[pts["record_id"].isin(list(focus))]
    for reaction in ("approve", "disapprove", "none"):
        sub = mine[mine["reaction"] == reaction]
        if sub.empty:
            continue
        fig.add_scatter(
            x=sub["semantic_x"], y=sub["semantic_y"], mode="markers",
            name=reaction.title(), customdata=sub["record_id"],
            marker=dict(color=REACTION_COLOR[reaction], size=9, opacity=0.95),
            hovertemplate="%{customdata}<br>“%{text}”<extra></extra>",
            text=sub["comment"].str.slice(0, 90))
    ctr = mine[mine["record_id"].isin(list(counter_ids))]
    if len(ctr):
        fig.add_scatter(x=ctr["semantic_x"], y=ctr["semantic_y"],
                        mode="markers", name="Counter-evidence",
                        marker=dict(size=15, color="rgba(0,0,0,0)",
                                    line=dict(color=C["red"], width=2)),
                        hoverinfo="skip")
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color=C["text"], size=11),
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(title="Semantic Dimension 1", showticklabels=False,
                   gridcolor=C["border"], zeroline=False),
        yaxis=dict(title="Semantic Dimension 2", showticklabels=False,
                   gridcolor=C["border"], zeroline=False))
    st.plotly_chart(fig, width="stretch", key=key)
    return True


def render_detail_point_panel(rid, an, g, dfp):
    """Source-linked detail for one clicked comment inside Theme Detail:
    original comment, provenance, theme relationship (Core/Related/Edge) with
    the measured similarity score, and the correction actions."""
    row = get_record(rid)
    if row is None:
        return
    st.markdown("**Selected comment** — every point remains linked to its "
                "source record.")
    comment_card(row, "tmap")  # original comment + reaction + IDs + evidence
    prow = dfp[dfp["record_id"] == rid]
    rel = prow["rel"].iloc[0] if len(prow) else "—"
    sim = float(prow["sim"].iloc[0]) if len(prow) else None
    rel_kind = {"Core": "human", "Related": "ai", "Edge": "review"}.get(
        rel, "gray")
    sim_txt = (f'<span class="ces-meta"> Similarity score: {sim:.3f} '
               f'(cosine to theme centroid — a measurable score, not an '
               f'objective truth)</span>' if sim is not None else "")
    st.markdown(pills((f"Relationship to Theme: {rel}", rel_kind)) + sim_txt,
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="ces-meta"><b>Current theme:</b> {g["name"]} · '
        f'<b>Dataset:</b> {row["dataset_id"]} · '
        f'<b>Activity:</b> {activity_name(row["activity_id"])}</div>',
        unsafe_allow_html=True)
    groups = theme_groups(an["analysis_id"])
    others = [x for x in groups if x["gid"] != g["gid"]]
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Remove from Theme", key=f"tm-rm1b-{rid}"):
            remove_records_from_group(g, [rid])
            st.rerun()
    with b2:
        with st.popover("Move to Another Theme"):
            if others:
                tgt = st.selectbox(
                    "Destination theme", [x["gid"] for x in others],
                    format_func=lambda gid: next(
                        x["name"] for x in others if x["gid"] == gid),
                    key=f"tm-mv1-{rid}")
                if st.button("Move", key=f"tm-mv1b-{rid}"):
                    remove_records_from_group(g, [rid])
                    add_records_to_group(
                        next(x for x in others if x["gid"] == tgt), [rid])
                    st.rerun()
            else:
                st.caption("No other theme exists yet.")
    with b3:
        with st.popover("Create Subtheme"):
            sn = st.text_input("Subtheme name", key=f"tm-sub1-{rid}")
            if st.button("Create", key=f"tm-sub1b-{rid}"):
                if sn.strip():
                    subs = g["obj"].setdefault("subthemes", [])
                    subs.append({"sid": f'{g["gid"]}-H{len(subs) + 1}',
                                 "name": sn.strip(), "record_ids": [rid],
                                 "keywords": [], "status": "confirmed",
                                 "source": "human"})
                    st.rerun()
    with b4:
        with st.popover("View Traceability"):
            chain = ("PROJECT\n  ↓\nENGAGEMENT ACTIVITY\n  ↓\nDATASET\n  ↓\n"
                     "RAW RECORD\n  ↓\nANALYSIS\n  ↓\nTHEME\n  ↓\n"
                     "THEME MAP POINT")
            st.markdown(f'<div class="ces-chain">{chain}</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="ces-meta"><b>Record:</b> {rid} · '
                f'<b>Response:</b> {row["response_id"]}<br>'
                f'<b>Source file:</b> {row["source_file"]}<br>'
                f'<b>Analysis:</b> {an["analysis_name"]} '
                f'({an["analysis_id"]})</div>', unsafe_allow_html=True)


DOMINANT_FILL = {"Mostly Approve": C["green"],
                 "Mostly Disapprove": C["red"],
                 "Mixed / Divided": C["yellow"]}


def render_theme_overview(an, groups, dim_label):
    """LEVEL 1 — THEME OVERVIEW. Answers: what themes exist and how do they
    relate to each other? Only themes are shown as points — never thousands
    of individual comments."""
    aid = an["analysis_id"]
    st.markdown("#### Theme Overview")
    st.markdown(
        '<div class="ces-note-ai">Nearby themes contain semantically related '
        'ideas. The axes are generated from patterns in the text and do not '
        'represent predefined planning concepts.</div>',
        unsafe_allow_html=True)
    vmode = st.radio("Overview view", ["Theme Map", "Reaction Spectrum"],
                     horizontal=True, key=f"tm-ovview-{aid}",
                     label_visibility="collapsed")
    adf = analysis_df(an)
    if vmode == "Reaction Spectrum":
        if adf is not None and len(adf):
            _reaction_spectrum(adf, groups, dim_label)
            st.caption("Which themes lean approval, which lean disapproval, "
                       "and which disagree internally — using only the "
                       "participant-provided reaction.")
        return

    # theme positions: centroid of member comments in the analysis's
    # semantic space — themes with related comments land near each other
    coords = get_semantic_coords(an, adf) if adf is not None else None
    rows = []
    for i, g in enumerate(groups):
        sub = records_for(g["record_ids"])
        cts = reaction_counts(sub) if len(sub) else \
            {"approve": 0, "disapprove": 0, "none": 0}
        label, fill, domword, share = dominant_reaction(cts)
        xs = ys = None
        if coords:
            xs = [coords["x"][r] for r in g["record_ids"] if r in coords["x"]]
            ys = [coords["y"][r] for r in g["record_ids"] if r in coords["y"]]
        if xs:
            x, y = float(np.mean(xs)), float(np.mean(ys))
        else:
            ang = 2 * np.pi * i / max(1, len(groups))
            x, y = float(np.cos(ang)), float(np.sin(ang))
        rows.append({"gid": g["gid"], "name": g["name"],
                     "status": g["status"], "n": len(g["record_ids"]),
                     "label": label, "domword": domword, "share": share,
                     "x": x, "y": y})
    tdf = pd.DataFrame(rows)
    nmax = max(1, int(tdf["n"].max()))
    tdf["size"] = 18 + 34 * np.sqrt(tdf["n"] / nmax)

    fig = go.Figure()
    for cat, fill in DOMINANT_FILL.items():
        s = tdf[tdf["label"] == cat]
        if s.empty:
            continue
        fig.add_scatter(
            x=s["x"], y=s["y"], mode="markers+text", name=cat,
            customdata=s["gid"],
            text=[f'{nm[:26]}<br>{n} comments<br>{sh:.0%} {dw}'
                  for nm, n, sh, dw in zip(s["name"], s["n"], s["share"],
                                           s["domword"])],
            textposition="top center",
            textfont=dict(size=10, color=C["text2"]),
            marker=dict(color=fill, size=s["size"].tolist(),
                        line=dict(color=[GROUP_STATUS_COLOR[t]
                                         for t in s["status"]], width=3)),
            hovertemplate="%{text}<extra></extra>")
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color=C["text"], size=12),
        legend=dict(orientation="h", y=1.09,
                    title="Dominant Participant Reaction"),
        xaxis=dict(title="Semantic Dimension 1", showgrid=True,
                   gridcolor=C["border"], zeroline=False,
                   showticklabels=False),
        yaxis=dict(title="Semantic Dimension 2", showgrid=True,
                   gridcolor=C["border"], zeroline=False,
                   showticklabels=False),
        dragmode="pan")
    sel = _selectable_chart(fig, f"tm-ov-{aid}")
    st.markdown(
        '<span class="ces-meta">Point size = number of comments · Fill = '
        'dominant participant reaction · Outline = theme status: </span>' +
        pills(*[(GROUP_STATUS_LABEL[s], k) for s, k in
                [("ai", "ai"), ("human", "human"),
                 ("validated", "validated")]]),
        unsafe_allow_html=True)
    if sel:
        gid = sel[0]
        if any(g["gid"] == gid for g in groups):
            st.session_state[f"tm-detail-{aid}"] = gid
            st.rerun()
    o1, o2 = st.columns([2.4, 1])
    pick = o1.selectbox(
        "Open a theme (or click its point above)",
        [g["gid"] for g in groups],
        format_func=lambda gid: next(
            f'{g["name"]} · {len(g["record_ids"])} comments '
            f'({GROUP_STATUS_LABEL[g["status"]]})'
            for g in groups if g["gid"] == gid),
        key=f"tm-ovpick-{aid}")
    if o2.button("Open Theme Detail", type="primary",
                 key=f"tm-ovopen-{aid}"):
        st.session_state[f"tm-detail-{aid}"] = pick
        st.rerun()


def _axis_title(axes_state, axis):
    e = axes_state.get(str(axis))
    if e and e.get("status") in ("accepted", "renamed"):
        return f'{e["neg"]}  ↔  {e["pos"]}'
    return f"Semantic Dimension {axis + 1}"


def render_theme_detail(an, g, dim_label):
    """LEVEL 2 — THEME DETAIL. Answers: how diverse are the comments inside
    this theme? The theme becomes its own analytical space (coordinates are
    recalculated within it), revealing subpositions, outliers, core vs edge
    membership, and possible subthemes."""
    aid = an["analysis_id"]
    gid = g["gid"]
    obj = g["obj"]
    if st.button("← Back to Theme Overview", key=f"tm-back-{aid}"):
        st.session_state[f"tm-detail-{aid}"] = None
        # clear the overview chart's persisted click so it doesn't
        # immediately re-open this theme
        st.session_state.pop(f"tm-ov-{aid}", None)
        st.rerun()

    sub = records_for(g["record_ids"])
    cts = reaction_counts(sub) if len(sub) else \
        {"approve": 0, "disapprove": 0, "none": 0}
    total = max(1, sum(cts.values()))
    with st.container(border=True):
        st.markdown(f"### {g['name']}")
        st.markdown(pills((GROUP_STATUS_LABEL[g["status"]], g["status"])),
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="ces-meta">{len(g["record_ids"])} comments · '
            f'Reaction: {cts["disapprove"]/total:.0%} Disapprove · '
            f'{cts["approve"]/total:.0%} Approve · '
            f'{cts["none"]/total:.0%} None · {dim_label}s: '
            f'{", ".join(sorted(sub["dim_value"].unique())) if len(sub) else "—"}'
            '</div>', unsafe_allow_html=True)
        if g["kind"] == "cluster":
            st.markdown(f'<div class="ces-note-ai"><b>AI Summary:</b> '
                        f'{obj["ai"]["summary"]}</div>', unsafe_allow_html=True)
        else:
            if obj.get("ai_original_summary"):
                st.markdown(f'<div class="ces-note-ai"><b>Original AI summary '
                            f'(preserved):</b> {obj["ai_original_summary"]}'
                            '</div>', unsafe_allow_html=True)
            if obj.get("interpretation"):
                st.markdown(f'<div class="ces-note-human"><b>Human '
                            f'interpretation:</b> {obj["interpretation"]}'
                            '</div>', unsafe_allow_html=True)
        with st.popover("Edit Theme Name"):
            new_name = st.text_input("Working name", g["name"],
                                     key=f"tm-edit-{gid}")
            if st.button("Apply", key=f"tm-editb-{gid}"):
                if g["kind"] == "cluster":
                    obj["ai"].setdefault("original_name", obj["ai"]["name"])
                    obj["ai"]["name"] = new_name
                else:
                    obj["name"] = new_name
                st.rerun()

    if sub.empty:
        st.caption("This theme has no comments left.")
        return
    space = theme_local_space(g)
    if space is None:
        st.caption("Not enough distinct comment text to build this theme's "
                   "landscape.")
        return
    dfp = space["df"].merge(
        sub[["record_id", "comment", "reaction", "dim_value", "dataset_id",
             "activity_id", "response_id", "source_file"]], on="record_id")
    maj = "approve" if cts["approve"] >= cts["disapprove"] else "disapprove"
    minr = "disapprove" if maj == "approve" else "approve"
    # positional/interpretive outliers: in the theme, but expressing a
    # materially different position (opposing the theme's majority reaction)
    dfp["interp_outlier"] = dfp["reaction"] == minr

    # ---- controls ----
    c1, c2 = st.columns([2, 1.2])
    amode = c1.radio("Axes", ["Semantic Landscape", "Interpretable Axes"],
                     horizontal=True, key=f"tm-amode-{gid}",
                     label_visibility="collapsed")
    show_outliers = c2.toggle(
        "SHOW OUTLIERS", value=True, key=f"tm-out-{gid}",
        help="Diamonds: semantic outliers (far from the theme's other "
             "comments). Red outer marker: interpretive outliers (a "
             "materially different position). Never hidden by opacity.")

    axes_state = st.session_state.setdefault("tm_axes", {}).setdefault(gid, {})
    if amode == "Interpretable Axes":
        st.markdown('<div class="ces-note-ai">AI can SUGGEST conceptual '
                    'dimensions based on differences among these comments. '
                    'Suggestions stay labeled AI Suggested until you accept, '
                    'rename, or reject them — the map is never silently '
                    'assigned conceptual meanings.</div>',
                    unsafe_allow_html=True)
        if st.button("Suggest axis interpretations",
                     key=f"tm-axsug-{gid}"):
            with st.spinner("Inspecting comments at the extremes of each "
                            "dimension…"):
                for ax in (0, 1):
                    sug = suggest_axis_interpretation(g, space, ax)
                    axes_state[str(ax)] = {**sug, "status": "ai"}
            st.rerun()
        for ax in (0, 1):
            e = axes_state.get(str(ax))
            if not e:
                continue
            with st.container(border=True):
                src = ("AI Suggested" if e["source"] == "llm"
                       else "Keyword-based suggestion (no AI key)")
                st.markdown(
                    pills((src, "ai") if e["status"] == "ai"
                          else ("Human Confirmed", "validated")) +
                    f' **Dimension {ax + 1}:** “{e["neg"]}” ↔ “{e["pos"]}”',
                    unsafe_allow_html=True)
                if e["status"] == "ai":
                    x1, x2, x3 = st.columns(3)
                    if x1.button("Accept Axis", key=f"tm-axok-{gid}-{ax}"):
                        e["status"] = "accepted"
                        st.rerun()
                    with x2.popover("Rename Axis"):
                        nn = st.text_input("Low end", e["neg"],
                                           key=f"tm-axn-{gid}-{ax}")
                        pp = st.text_input("High end", e["pos"],
                                           key=f"tm-axp-{gid}-{ax}")
                        if st.button("Apply names",
                                     key=f"tm-axrn-{gid}-{ax}"):
                            e.update({"neg": nn.strip() or e["neg"],
                                      "pos": pp.strip() or e["pos"],
                                      "status": "renamed"})
                            st.rerun()
                    if x3.button("Reject Axis", key=f"tm-axrej-{gid}-{ax}"):
                        axes_state.pop(str(ax), None)
                        st.rerun()
    else:
        st.markdown(f'<div class="ces-meta">{SEMANTIC_AXES_NOTE}</div>',
                    unsafe_allow_html=True)

    # ---- comment landscape ----
    fig = go.Figure()
    for rel in ("Core", "Related", "Edge"):
        s = dfp[dfp["rel"] == rel]
        if s.empty:
            continue
        symbols = np.where(s["sem_outlier"] & show_outliers,
                           "diamond", "circle")
        fig.add_scatter(
            x=s["x"], y=s["y"], mode="markers", name=rel,
            customdata=s["record_id"],
            marker=dict(color=REL_COLOR[rel], size=12,
                        symbol=symbols.tolist(),
                        line=dict(color=s["reaction"].map(
                            REACTION_COLOR).tolist(), width=2)),
            hovertemplate=("%{customdata} · " + rel +
                           "<br>“%{text}”<extra></extra>"),
            text=s["comment"].str.slice(0, 90))
    if show_outliers:
        io = dfp[dfp["interp_outlier"]]
        if len(io):
            fig.add_scatter(
                x=io["x"], y=io["y"], mode="markers",
                name="Interpretive outlier",
                customdata=io["record_id"],
                marker=dict(size=19, color="rgba(0,0,0,0)",
                            line=dict(color=C["red"], width=2)),
                hovertemplate=("Interpretive outlier: %{customdata}"
                               "<extra></extra>"))
    for stobj in obj.get("subthemes", []):
        sp = dfp[dfp["record_id"].isin(stobj["record_ids"])]
        if len(sp) < 1:
            continue
        color = C["purple"] if stobj["status"] == "ai" else C["blue"]
        fig.add_annotation(
            x=float(sp["x"].median()), y=float(sp["y"].median()),
            text=stobj["name"][:30], showarrow=False,
            font=dict(size=10, color=color),
            bgcolor="rgba(255,255,255,0.75)", bordercolor=color,
            borderwidth=1)
    fig.update_layout(
        height=540, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color=C["text"], size=12),
        legend=dict(orientation="h", y=1.09, title="Relationship to Theme"),
        xaxis=dict(title=_axis_title(axes_state, 0), showgrid=True,
                   gridcolor=C["border"], zeroline=False,
                   showticklabels=False),
        yaxis=dict(title=_axis_title(axes_state, 1), showgrid=True,
                   gridcolor=C["border"], zeroline=False,
                   showticklabels=False),
        dragmode="lasso")
    selected = _selectable_chart(fig, f"tm-dt-{gid}")

    def _dot(color):
        return (f'<span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:50%;background:{color};margin:0 4px 0 10px;">'
                '</span>')

    def _ring(color):
        return (f'<span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:50%;background:#fff;border:2.5px solid '
                f'{color};margin:0 4px 0 10px;"></span>')

    st.markdown(
        '<div class="ces-meta"><b>COMMENT RELATIONSHIP (fill)</b>'
        + _dot(C["blue"]) + 'Core' + _dot(C["purple"]) + 'Related'
        + _dot(C["yellow"]) + 'Edge'
        + ' &nbsp;·&nbsp; <b>REACTION (border)</b>'
        + _ring(C["green"]) + 'Approve' + _ring(C["red"]) + 'Disapprove'
        + _ring(C["yellow"]) + 'None'
        + ' &nbsp;·&nbsp; ◇ semantic outlier · '
        + _ring(C["red"]) + 'interpretive outlier</div>',
        unsafe_allow_html=True)
    st.caption("Fill = how strongly the comment belongs to the theme "
               "(measured similarity, in tertiles — the thresholds are "
               "analytical conveniences, not objective truths). "
               "Border = participant reaction. Click a point to inspect it; "
               "lasso-select to edit the grouping.")

    # ---- selection & correction ----
    label_map = {rid: f'{rid} · {c[:70]}'
                 for rid, c in zip(dfp["record_id"], dfp["comment"])}
    sel_key, last_key = f"tm2-sel-{gid}", f"tm2-lastsel-{gid}"
    if selected and selected != st.session_state.get(last_key):
        st.session_state[sel_key] = [r for r in selected if r in label_map]
        st.session_state[last_key] = selected
    st.session_state[sel_key] = [r for r in st.session_state.get(sel_key, [])
                                 if r in label_map]
    manual = st.multiselect(
        "Selected comments (from map selection, or pick manually)",
        dfp["record_id"].tolist(),
        format_func=lambda r: label_map.get(r, r), key=sel_key)
    groups = theme_groups(aid)
    others = [x for x in groups if x["gid"] != gid]
    if len(manual) == 1:
        render_detail_point_panel(manual[0], an, g, dfp)
    elif len(manual) > 1:
        st.caption(f"{len(manual)} comments selected.")
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            if st.button("REMOVE SELECTED FROM THEME",
                         key=f"tm-rmsel-{gid}"):
                remove_records_from_group(g, manual)
                st.toast("Removed — counts recalculated from record IDs")
                st.rerun()
        with cc2:
            with st.popover("MOVE SELECTED TO ANOTHER THEME"):
                if others:
                    tgt = st.selectbox(
                        "Destination theme", [x["gid"] for x in others],
                        format_func=lambda k: next(
                            x["name"] for x in others if x["gid"] == k),
                        key=f"tm-mvsel-{gid}")
                    if st.button("Move selected", key=f"tm-mvselb-{gid}"):
                        remove_records_from_group(g, manual)
                        add_records_to_group(
                            next(x for x in others if x["gid"] == tgt),
                            manual)
                        st.rerun()
                else:
                    st.caption("No other theme exists yet.")
        with cc3:
            with st.popover("CREATE SUBTHEME FROM SELECTED"):
                sn = st.text_input("Subtheme name", key=f"tm-subn-{gid}")
                if st.button("Create subtheme", key=f"tm-subnb-{gid}"):
                    if sn.strip():
                        subs = obj.setdefault("subthemes", [])
                        subs.append({"sid": f"{gid}-H{len(subs) + 1}",
                                     "name": sn.strip(),
                                     "record_ids": sorted(manual),
                                     "keywords": [], "status": "confirmed",
                                     "source": "human"})
                        st.rerun()
        with cc4:
            with st.popover("CREATE NEW THEME FROM SELECTED"):
                tn = st.text_input("Theme name (short)", key=f"tm-nt-{gid}")
                ti = st.text_area("Interpretation", key=f"tm-nti-{gid}")
                if st.button("Create theme", key=f"tm-ntb-{gid}"):
                    if not tn.strip():
                        st.error("A theme needs a name.")
                    else:
                        nsub = records_for(manual)
                        prov = provenance_from_records(manual)
                        theme_id = next_id("theme_seq", "TH-")
                        st.session_state.themes.append({
                            "theme_id": theme_id, "origin": "human",
                            "analysis_id": aid,
                            "dimension": an["comparison_dimension"]["name"],
                            "ai_original_name": None,
                            "ai_original_summary": None, "ai_source": None,
                            "name": tn.strip(),
                            "interpretation": ti.strip(),
                            "record_ids": sorted(manual),
                            "excluded_record_ids": [],
                            "counts": reaction_counts(nsub),
                            "n_respondents":
                                int(nsub["response_id"].nunique()),
                            "tags": [], "notes": "Created from Theme Map "
                                                 "selection",
                            "counter_ids": [], "constraints": [],
                            "validated":
                                datetime.date.today().isoformat(),
                            "status": "HUMAN VALIDATED",
                            "cluster_key": None, **prov})
                        st.toast(f"Human theme {theme_id} created from "
                                 f"{len(manual)} selected comments — the "
                                 "Theme Overview reflects the change")
                        st.rerun()

    # ---- subthemes ----
    st.markdown("---")
    st.markdown("**Subthemes** — AI suggests; you accept, rename, merge, or "
                "reject. Suggested boundaries are annotations, not facts.")
    if st.button("Suggest Subthemes", key=f"tm-subsug-{gid}",
                 disabled=(len(sub) < 6)):
        with st.spinner("Looking for internal structure…"):
            confirmed = [s for s in obj.get("subthemes", [])
                         if s["status"] == "confirmed"]
            obj["subthemes"] = confirmed + suggest_subthemes(g, an)
        st.rerun()
    subthemes = obj.get("subthemes", [])
    if not subthemes:
        st.caption("No subthemes yet.")
    for stobj in subthemes:
        with st.container(border=True):
            st.markdown(
                pills(("AI Suggested Subtheme", "ai")
                      if stobj["status"] == "ai"
                      else ("Human Confirmed Subtheme", "human")) +
                f' **{stobj["name"]}** '
                f'<span class="ces-meta">{len(stobj["record_ids"])} comments'
                + (f' · {", ".join(stobj["keywords"][:4])}'
                   if stobj.get("keywords") else "") + "</span>",
                unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            if stobj["status"] == "ai":
                if s1.button("Accept Subtheme",
                             key=f"tm-stok-{stobj['sid']}"):
                    stobj["status"] = "confirmed"
                    st.rerun()
            with s2.popover("Rename"):
                nn = st.text_input("Name", stobj["name"],
                                   key=f"tm-strn-{stobj['sid']}")
                if st.button("Apply", key=f"tm-strnb-{stobj['sid']}"):
                    stobj["name"] = nn.strip() or stobj["name"]
                    stobj["status"] = "confirmed"
                    st.rerun()
            with s3.popover("Merge"):
                targets = [x for x in subthemes if x["sid"] != stobj["sid"]]
                if targets:
                    tgt = st.selectbox(
                        "Merge into", [x["sid"] for x in targets],
                        format_func=lambda k: next(
                            x["name"] for x in targets if x["sid"] == k),
                        key=f"tm-stmg-{stobj['sid']}")
                    if st.button("Merge subthemes",
                                 key=f"tm-stmgb-{stobj['sid']}"):
                        t = next(x for x in targets if x["sid"] == tgt)
                        t["record_ids"] = sorted(set(t["record_ids"])
                                                 | set(stobj["record_ids"]))
                        t["status"] = "confirmed"
                        obj["subthemes"] = [x for x in subthemes
                                            if x["sid"] != stobj["sid"]]
                        st.rerun()
                else:
                    st.caption("No other subtheme to merge with.")
            if s4.button("Reject", key=f"tm-strej-{stobj['sid']}"):
                obj["subthemes"] = [x for x in subthemes
                                    if x["sid"] != stobj["sid"]]
                st.rerun()

    # ---- validation: an explicit review of the diversity inside the theme --
    st.markdown("---")
    if g["kind"] == "cluster":
        st.markdown("#### Validation Review")
        rel_counts = dfp["rel"].value_counts()
        n_sem = int((dfp["sem_outlier"]).sum())
        n_int = int(dfp["interp_outlier"].sum())
        removed = obj.get("removed_record_ids", [])
        with st.container(border=True):
            st.markdown(
                f'<div class="ces-meta" style="line-height:1.9;">'
                f'<b>Theme:</b> {g["name"]} · '
                f'<b>Human interpretation:</b> not written yet<br>'
                f'<b>Total comments:</b> {len(dfp)} · '
                f'<b>Core:</b> {int(rel_counts.get("Core", 0))} · '
                f'<b>Related:</b> {int(rel_counts.get("Related", 0))} · '
                f'<b>Edge:</b> {int(rel_counts.get("Edge", 0))}<br>'
                f'<b>Outliers:</b> {n_sem} semantic · {n_int} interpretive · '
                f'<b>Counter-evidence:</b> {len(obj.get("counter_ids", []))}'
                f'<br><b>Approve / Disapprove / None:</b> {cts["approve"]} / '
                f'{cts["disapprove"]} / {cts["none"]}<br>'
                f'<b>Subthemes:</b> '
                f'{", ".join(s["name"] for s in subthemes) or "none"}<br>'
                f'<b>Comments included:</b> {len(g["record_ids"])} · '
                f'<b>Comments removed during review:</b> {len(removed)}'
                '</div>', unsafe_allow_html=True)
        v1, v2 = st.columns([1, 1])
        validating_here = (st.session_state.validating_cluster == gid
                           and st.session_state.get("validating_in_map"))
        if not validating_here:
            if v1.button("VALIDATE THEME", type="primary",
                         key=f"tm-val-{gid}"):
                st.session_state.validating_cluster = gid
                st.session_state.validating_in_map = True
                st.rerun()
            if v2.button("Reject Theme", key=f"tm-rej-{gid}"):
                obj["status"] = "rejected"
                st.session_state[f"tm-detail-{aid}"] = None
                st.session_state.pop(f"tm-ov-{aid}", None)
                st.rerun()
        else:
            render_validation_form(obj)
    else:
        st.markdown(pills(("Human Validated", "validated")) +
                    '<span class="ces-meta"> This theme has been validated — '
                    'membership edits here recalculate its counts and '
                    'provenance, and the Evidence Library reflects them.'
                    '</span>', unsafe_allow_html=True)


def render_theme_map(an, view, dim_label):
    """THEME MAP module — a TWO-LEVEL visual exploration.
    LEVEL 1 (Theme Overview): what themes exist and how do they relate?
    LEVEL 2 (Theme Detail): how diverse are the comments inside one theme?
    The two questions are never combined into one crowded visualization.
    AI suggests a grouping → human sees it → inspects original comments →
    modifies the grouping → validates the theme."""
    aid = an["analysis_id"]
    groups = theme_groups(aid)
    detail_key = f"tm-detail-{aid}"
    pending = st.session_state.pop("theme_map_focus", None)
    if pending and any(g["gid"] == pending for g in groups):
        st.session_state[detail_key] = pending
    if not groups:
        st.markdown('<div class="ces-note-human">No themes yet — run '
                    '<b>Run Thematic Analysis</b> in the Themes tab to let '
                    'AI suggest clusters, then explore and correct them '
                    'here.</div>', unsafe_allow_html=True)
        return
    gid = st.session_state.get(detail_key)
    g = next((x for x in groups if x["gid"] == gid), None) if gid else None
    if gid and g is None:
        # the theme was validated, split, or rejected — the overview
        # reflects the updated human interpretation
        st.session_state[detail_key] = None
        st.session_state.pop(f"tm-ov-{aid}", None)
    if g is not None:
        render_theme_detail(an, g, dim_label)
    else:
        render_theme_overview(an, groups, dim_label)


def page_insights():
    st.title("Insights Playground")
    st.markdown(
        pills(("AI-assisted analysis", "ai")) +
        f'<span style="color:{C["text2"]};font-size:13px;"> AI suggestions are '
        'starting points for human interpretation.</span>', unsafe_allow_html=True)

    if not st.session_state.analyses:
        st.markdown('<div class="ces-note-human">The Playground is generated '
                    'from an <b>Analysis</b>, not directly from uploaded files. '
                    'Go to <b>02 Analysis Setup</b>, define what you want to '
                    'learn, and press <b>Generate Playground</b>.</div>',
                    unsafe_allow_html=True)
        return

    # ---------- ACTIVE ANALYSIS ----------
    an_ids = [a["analysis_id"] for a in st.session_state.analyses]
    cur = st.session_state.active_analysis_id
    idx = an_ids.index(cur) if cur in an_ids else 0
    sel = st.selectbox("Analysis", an_ids, index=idx,
                       format_func=lambda a: f"{a} · {analysis_name(a)}")
    st.session_state.active_analysis_id = sel
    an = get_analysis(sel)
    adf = analysis_df(an)
    if adf is None or adf.empty:
        st.markdown('<div class="ces-note-warn">The source data for this '
                    'analysis is no longer processed. Re-process the activity '
                    'in <b>01 Data + Context</b>.</div>', unsafe_allow_html=True)
        return

    dim_name = an["comparison_dimension"]["name"]
    dim_label = ("Group" if dim_name == COMBINED_DIMENSION
                 else dim_name.split(" /")[0])
    caps = an.get("capabilities", {})
    activity = get_activity(an["activity_id"])
    enabled = [m for m in an["enabled_modules"] if m in MODULE_DEFS]

    # ---------- ANALYSIS HEADER ----------
    with st.container(border=True):
        st.markdown(
            f'<div class="ces-meta" style="line-height:1.9;">'
            f'<b>Analysis:</b> {an["analysis_name"]} ({an["analysis_id"]})<br>'
            f'<b>Engagement Activity:</b> {activity_name(an["activity_id"])}<br>'
            f'<b>Datasets:</b> '
            f'{", ".join(an["comparison_dimension"]["values"])}<br>'
            f'<b>Unit of Analysis:</b> {an["unit_of_analysis"]} · '
            f'<b>Compare by:</b> {dim_label if dim_name != COMBINED_DIMENSION else "— (combined)"}<br>'
            f'<b>Purpose:</b> {an["purpose"] or "—"}</div>',
            unsafe_allow_html=True)
        if an["questions"]:
            with st.expander(
                    f"Questions being investigated ({len(an['questions'])})"):
                for q in an["questions"]:
                    st.markdown(f"- {q}")
        if an.get("constraint_ids"):
            st.markdown("Confirmed relevant constraints: " + "".join(
                pill(f'{cid} · '
                     f'{next((c["name"] for c in st.session_state.constraints if c["id"] == cid), cid)}',
                     "validated") for cid in an["constraint_ids"]),
                unsafe_allow_html=True)

    # constraints drawer — always available in the playground
    with st.sidebar.expander("☷ Project Constraints", expanded=False):
        if not st.session_state.constraints:
            st.caption("None documented yet — add them in Data + Context.")
        for con in st.session_state.constraints:
            st.markdown(f"**{con['id']} — {con['name']}**")
            st.markdown(pills((con["type"], "gray"), (con["status"], "validated")),
                        unsafe_allow_html=True)
            if con["description"]:
                st.caption(con["description"][:160])

    # ---------- ANALYSIS-AWARE FILTER BAR (derived from selected data) ----------
    ds_opts = {d["dataset_id"]: (d["dataset_name"] or d["source_file"])
               for d in activity["datasets"]
               if d["dataset_id"] in an["dataset_ids"] and d["df"] is not None}
    show_reaction = bool(caps.get("reaction", True))
    fcols = st.columns([1.4, 1.2, 1, 1.2, 1.4] if show_reaction
                       else [1.4, 1.2, 1.2, 1.4])
    f_ds = fcols[0].selectbox("Dataset", ["All"] + list(ds_opts),
                              format_func=lambda k: ds_opts.get(k, k))
    f_dim = fcols[1].selectbox(dim_label,
                               ["All"] + sorted(adf["dim_value"].unique()))
    fi = 2
    if show_reaction:
        f_reac = fcols[fi].selectbox("Reaction",
                                     ["All", "approve", "disapprove", "none"])
        fi += 1
    else:
        f_reac = "All"
    theme_opts = (["All"] +
                  [f'{c["key"]} · {c["ai"]["name"]}'
                   for c in analysis_clusters(sel).values()] +
                  [f"tag:{t}" for t in all_known_tags()])
    f_theme = fcols[fi].selectbox("Theme / Tag", theme_opts)
    f_kw = fcols[fi + 1].text_input("Keyword", placeholder="Search comments...")

    view = adf
    if f_ds != "All":
        view = view[view["dataset_id"] == f_ds]
    if f_dim != "All":
        view = view[view["dim_value"] == f_dim]
    if f_reac != "All":
        view = view[view["reaction"] == f_reac]
    if f_theme != "All":
        if f_theme.startswith("tag:"):
            t = f_theme[4:]
            rids = [rid for rid, entries in st.session_state.tags.items()
                    if any(e["tag"] == t for e in entries)]
            view = view[view["record_id"].isin(rids)]
        else:
            ckey = f_theme.split(" · ")[0]
            cl = st.session_state.clusters.get(ckey)
            if cl:
                view = view[view["record_id"].isin(cl["record_ids"])]
    if f_kw.strip():
        view = view[view["comment"].str.contains(re.escape(f_kw.strip()),
                                                 case=False, na=False)]

    st.caption(f"{an['analysis_name']} · {len(view)} comments · "
               f"{view['response_id'].nunique()} unique response IDs match the "
               f"current filters (analysis total: {len(adf)}).")

    # ---------- MODULES (only enabled + supported ones are shown) ----------
    tabs = st.tabs([MODULE_DEFS[m][0] for m in enabled])
    module_tabs = dict(zip(enabled, tabs))
    tab_over = module_tabs.get("overview")
    tab_comments = module_tabs.get("comments")
    tab_themes = module_tabs.get("themes")
    tab_compare = module_tabs.get("compare")

    # ---------------- OVERVIEW ----------------
    if tab_over is not None:
        with tab_over:
            m = st.columns(4)
            m[0].metric("Analysis comments", len(adf))
            m[1].metric("Filtered comments", len(view))
            m[2].metric("Datasets", len(ds_opts))
            m[3].metric("Unique response IDs", adf["response_id"].nunique())
            if show_reaction:
                st.subheader(f"Participant Reaction by {dim_label}")
                st.caption("The reaction field supplied in the dataset is the "
                           "authoritative reaction variable. This is not AI "
                           "sentiment.")
                st.plotly_chart(
                    reaction_chart(view if len(view) else adf,
                                   group_label=dim_label), width="stretch")
                st.dataframe(group_pct_table(view if len(view) else adf,
                                             dim_label),
                             width="stretch", hide_index=True)
            else:
                st.subheader(f"Comments by {dim_label}")
                st.caption("No reaction variable exists in these datasets — "
                           "showing deterministic counts only.")
                src = view if len(view) else adf
                per = src.groupby("dim_value").agg(
                    comments=("record_id", "count"),
                    unique_response_ids=("response_id", "nunique")
                ).reset_index().rename(columns={"dim_value": dim_label})
                st.dataframe(per, width="stretch", hide_index=True)
            if st.button("Save this breakdown as a Quantitative Pattern"):
                src = view if len(view) else adf
                add_evidence({
                    "type": "Quantitative Pattern",
                    "record_ids": src["record_id"].tolist(),
                    "dim_value": ", ".join(sorted(src["dim_value"].unique())),
                    "reaction": None, "counts": reaction_counts(src),
                    "per_group": {s: reaction_counts(src[src["dim_value"] == s])
                                  for s in sorted(src["dim_value"].unique())},
                    "original_comment": None, "selected_quote": None,
                    "tags": [], "theme_id": None, "status": "Human Selected",
                    **provenance_from_records(src["record_id"].tolist())})
                st.toast("Quantitative pattern saved (Unassigned — assign it "
                         "to a theme in the Evidence Library)")

    # ---------------- COMMENTS ----------------
    if tab_comments is not None:
        with tab_comments:
            with st.expander("Bulk tagging (select multiple comments)"):
                label_map = {rid: f'{rid} · {c[:70]}'
                             for rid, c in zip(view["record_id"], view["comment"])}
                bsel = st.multiselect("Comments", view["record_id"].tolist(),
                                      format_func=lambda r: label_map.get(r, r))
                bt1, bt2 = st.columns(2)
                btag = bt1.selectbox("Tag", ["—"] + all_known_tags(),
                                     key="bulk-tag-sel")
                bnew = bt2.text_input("Or new tag", key="bulk-tag-new")
                if st.button("Apply tag to selected"):
                    chosen = bnew.strip() or (btag if btag != "—" else "")
                    if chosen and bsel:
                        for rid in bsel:
                            add_tag(rid, chosen, "human")
                        st.rerun()
            page_size = 15
            n_pages = max(1, (len(view) - 1) // page_size + 1)
            pg = st.number_input("Page", 1, n_pages, 1) if n_pages > 1 else 1
            for _, r in view.iloc[(pg - 1) * page_size: pg * page_size].iterrows():
                comment_card(r, "cx")
                memberships = [c["ai"]["name"] for c in
                               analysis_clusters(sel).values()
                               if r["record_id"] in c["record_ids"]]
                if memberships:
                    st.caption("In themes: " + " · ".join(memberships))

    # ---------------- THEMES ----------------
    if tab_themes is not None:
        with tab_themes:
            provider, _ = llm_provider()
            if provider is None:
                st.markdown('<div class="ces-note-warn">AI theme interpretation '
                            'unavailable. Configure an API key (ANTHROPIC_API_KEY or '
                            'OPENAI_API_KEY) to enable AI-generated theme names and '
                            'summaries. Local clustering still runs; all human '
                            'workflows remain available.</div>',
                            unsafe_allow_html=True)

            st.markdown("**Analyze:**")
            scope = st.radio(
                "Analysis scope",
                ["Current Dataset", "Selected Datasets", "All Analysis Datasets"],
                index=2, horizontal=True, label_visibility="collapsed")
            if scope == "Current Dataset":
                if f_ds == "All":
                    st.caption("Select a single dataset in the filter bar above, "
                               "or choose a wider scope.")
                    scope_df = adf
                    scope_desc = f"{an['analysis_name']} (all datasets)"
                else:
                    scope_df = adf[adf["dataset_id"] == f_ds]
                    scope_desc = f"{ds_opts.get(f_ds, f_ds)}"
            elif scope == "Selected Datasets":
                sel_ds = st.multiselect("Datasets to analyze", list(ds_opts),
                                        default=list(ds_opts),
                                        format_func=lambda k: ds_opts[k])
                scope_df = adf[adf["dataset_id"].isin(sel_ds)]
                scope_desc = (f"{an['analysis_name']} — {len(sel_ds)} "
                              "selected datasets")
            else:
                scope_df = adf
                scope_desc = f"{an['analysis_name']} (all analysis datasets)"

            c1, c2 = st.columns([1, 3])
            if c1.button("Run Thematic Analysis", type="primary",
                         disabled=(len(scope_df) == 0)):
                with st.spinner(f"Clustering per {dim_label.lower()} "
                                "(TF-IDF + KMeans)"
                                + (" and asking the LLM to interpret each "
                                   "cluster…" if provider else "…")):
                    run_thematic_analysis(an, scope_df, scope_desc)
                st.rerun()
            c2.caption(f"Scope: {scope_desc} · {len(scope_df)} comments. "
                       f"Comments are clustered per {dim_label.lower()} as "
                       "configured by this analysis; dataset and activity "
                       "provenance is preserved inside every cluster and theme.")

            render_create_human_theme(adf)

            if st.session_state.validating_cluster and \
                    not st.session_state.get("validating_in_map"):
                cl = st.session_state.clusters.get(
                    st.session_state.validating_cluster)
                if cl and cl.get("analysis_id") == sel:
                    render_validation_form(cl)
            an_clusters = analysis_clusters(sel)
            if an.get("cluster_run_done"):
                st.caption(f"Showing clusters from: "
                           f"{an.get('cluster_scope_desc', '')}")
                for grp in sorted({c["group"] for c in an_clusters.values()}):
                    gcl = [c for c in an_clusters.values() if c["group"] == grp]
                    if gcl:
                        st.subheader(grp)
                        for cl in gcl:
                            render_cluster_card(cl)
            elif not st.session_state.themes:
                st.caption("No thematic analysis yet — press Run Thematic "
                           "Analysis.")

    # ---------------- COMPARE ----------------
    if tab_compare is not None:
        with tab_compare:
            st.markdown(f'### COMPARE BY: {dim_label}')
            st.markdown(pills(*[(v, "gray") for v in
                                an["comparison_dimension"]["values"]]),
                        unsafe_allow_html=True)
            if show_reaction:
                st.subheader("Participant Reaction")
                st.plotly_chart(reaction_chart(adf, group_label=dim_label),
                                width="stretch", key="compare-chart")
                st.dataframe(group_pct_table(adf, dim_label), width="stretch",
                             hide_index=True)
            an_clusters = analysis_clusters(sel)
            st.subheader(f"Themes by {dim_label}")
            if not an.get("cluster_run_done"):
                st.caption("Run the thematic analysis in the Themes tab first.")
            else:
                grps = sorted({c["group"] for c in an_clusters.values()})
                grp_cols = st.columns(max(1, len(grps)))
                for i, grp in enumerate(grps):
                    with grp_cols[i]:
                        st.markdown(f"**{grp}**")
                        for c in [c for c in an_clusters.values()
                                  if c["group"] == grp]:
                            n = max(1, c["n_comments"])
                            with st.container(border=True):
                                kind = ("validated" if c["status"] == "validated"
                                        else "ai")
                                st.markdown(pills((c["ai"]["name"][:38], kind)),
                                            unsafe_allow_html=True)
                                st.markdown(
                                    f'<div class="ces-meta">{c["n_comments"]} comments'
                                    f'<br>Approve {c["counts"]["approve"]/n:.0%} · '
                                    f'Disapprove {c["counts"]["disapprove"]/n:.0%}<br>'
                                    f'{" · ".join(c["keywords"][:4])}</div>',
                                    unsafe_allow_html=True)
                st.subheader(f"Patterns Across {dim_label}s")
                st.caption("AI-computed proposals from keyword overlap between "
                           "clusters. All linked to the same engagement activity; "
                           "proposals remain proposals until reviewed by a human.")
                for p in cross_group_patterns(an):
                    reviewed = st.session_state.cross_reviews.get(p["key"])
                    with st.container(border=True):
                        st.markdown(pills(
                            (p["relationship"], "ai"),
                            *([("Reviewed", "validated")] if reviewed else [])),
                            unsafe_allow_html=True)
                        names = " + ".join(f'{c["ai"]["name"]} ({c["group"]})'
                                           for c in p["clusters"])
                        st.markdown(f"**{names}**")
                        st.markdown(f'<div class="ces-meta">Shared keywords: '
                                    f'{", ".join(p["shared_keywords"][:6]) or "—"} · '
                                    f'Activity: '
                                    f'{activity_name(p["clusters"][0]["activity_ids"][0]) if p["clusters"][0]["activity_ids"] else "—"}'
                                    '</div>', unsafe_allow_html=True)
                        with st.expander("Supporting comments"):
                            for c in p["clusters"]:
                                for rid in c["rep_ids"][:2]:
                                    r = get_record(rid)
                                    if r is not None:
                                        comment_card(r, f"xs-sup-{p['key']}",
                                                     show_actions=False)
                        ctr = [rid for c in p["clusters"]
                               for rid in c["counter_ids"][:2]]
                        if ctr:
                            with st.expander("Contradictory comments"):
                                for rid in ctr:
                                    r = get_record(rid)
                                    if r is not None:
                                        comment_card(r, f"xs-ctr-{p['key']}",
                                                     show_actions=False)
                        if not reviewed:
                            if st.button("Review — mark as human-reviewed",
                                         key=f"xsrev-{p['key']}"):
                                st.session_state.cross_reviews[p["key"]] = {
                                    "date": datetime.date.today().isoformat()}
                                st.rerun()
                        else:
                            if st.button(f"Save as Cross-{dim_label} Pattern "
                                         "(evidence)",
                                         key=f"xssave-{p['key']}"):
                                rids = sorted({rid for c in p["clusters"]
                                               for rid in c["record_ids"]})
                                sub = records_for(rids)
                                add_evidence({
                                    "type": "Cross-Group Pattern",
                                    "record_ids": rids,
                                    "dim_value": ", ".join(
                                        sorted(sub["dim_value"].unique())),
                                    "reaction": None,
                                    "counts": reaction_counts(sub),
                                    "original_comment": None,
                                    "selected_quote": None,
                                    "pattern_label": f'{p["relationship"]}: ' +
                                    " + ".join(c["ai"]["name"]
                                               for c in p["clusters"]),
                                    "tags": p["shared_keywords"][:5],
                                    "theme_id": None, "status": "Human Reviewed",
                                    **provenance_from_records(rids)})
                                st.toast(f"Cross-{dim_label.lower()} pattern "
                                         "saved (Unassigned)")

    # ---------------- MAP (only when coordinates exist) ----------------
    if "map" in module_tabs:
        with module_tabs["map"]:
            st.subheader("Spatial Distribution")
            if "lat" in view.columns and "lon" in view.columns:
                pts = view.dropna(subset=["lat", "lon"])
                pts = pts[pd.to_numeric(pts["lat"], errors="coerce").notna()
                          & pd.to_numeric(pts["lon"], errors="coerce").notna()]
                if len(pts):
                    st.map(pts.assign(lat=pd.to_numeric(pts["lat"]),
                                      lon=pd.to_numeric(pts["lon"]))
                           [["lat", "lon"]])
                    st.caption(f"{len(pts)} records with coordinates "
                               "(current filters applied).")
                else:
                    st.caption("No records with valid coordinates match the "
                               "current filters.")
            else:
                st.caption("No coordinate fields in the current view.")

    # ---------------- TIMELINE (only when dates exist) ----------------
    if "timeline" in module_tabs:
        with module_tabs["timeline"]:
            st.subheader("Comments Over Time")
            if "record_date" in view.columns:
                tl = view.copy()
                tl["record_date"] = pd.to_datetime(tl["record_date"],
                                                   errors="coerce")
                tl = tl.dropna(subset=["record_date"])
                if len(tl):
                    per = tl.groupby([pd.Grouper(key="record_date", freq="W"),
                                      "dim_value"]).size().reset_index(name="comments")
                    fig = go.Figure()
                    for grp in sorted(per["dim_value"].unique()):
                        g = per[per["dim_value"] == grp]
                        fig.add_scatter(x=g["record_date"], y=g["comments"],
                                        name=grp, mode="lines+markers")
                    fig.update_layout(height=340, plot_bgcolor="#FFFFFF",
                                      paper_bgcolor="#FFFFFF",
                                      font=dict(color=C["text"], size=13),
                                      legend=dict(title=dim_label))
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.caption("No parseable dates in the current view.")
            else:
                st.caption("No date fields in the current view.")

    # ------------- RANKINGS / STAKEHOLDERS (capability-gated) -------------
    if "rankings" in module_tabs:
        with module_tabs["rankings"]:
            st.subheader("Ranking Analysis")
            rank_cols = [c for c in view.columns
                         if any(h in c for h in _RANK_HINTS)
                         and pd.api.types.is_numeric_dtype(view[c])]
            if rank_cols:
                agg = view.groupby("dim_value")[rank_cols].mean().round(2)
                agg.index.name = dim_label
                st.dataframe(agg.reset_index(), width="stretch",
                             hide_index=True)
                st.caption("Mean values of detected ranking fields, by "
                           f"{dim_label.lower()}. Deterministic Python "
                           "calculations.")
            else:
                st.caption("No numeric ranking fields in the current view.")

    # ---------------- THEME MAP (semantic inspection & correction) ----------
    if "theme_map" in module_tabs:
        with module_tabs["theme_map"]:
            st.subheader("Theme Map")
            st.caption("A two-level exploration: the Theme Overview shows "
                       "themes as points (what themes exist and how they "
                       "relate); clicking a theme opens its Theme Detail "
                       "(how diverse the comments inside it are). Spatial "
                       "proximity indicates semantic similarity — not "
                       "geographic proximity or causal relationships.")
            render_theme_map(an, view, dim_label)

    if "stakeholders" in module_tabs:
        with module_tabs["stakeholders"]:
            st.subheader("Stakeholder / Demographic Comparison")
            demo_cols = [c for c in view.columns
                         if any(h in c for h in _DEMO_HINTS)]
            if demo_cols:
                dcol = st.selectbox("Demographic field", demo_cols)
                per = view.groupby([dcol, "reaction"]).size().reset_index(
                    name="comments") if show_reaction else \
                    view.groupby(dcol).size().reset_index(name="comments")
                st.dataframe(per, width="stretch", hide_index=True)
            else:
                st.caption("No demographic fields in the current view.")


# ----------------------------------------------------------------------------
# PAGE 04 — LIBRARIES  (theme-centric Evidence Library)
# ----------------------------------------------------------------------------

def evidence_snippet(ev):
    if ev.get("selected_quote"):
        return f'“…{ev["selected_quote"][:90]}…”'
    if ev.get("original_comment"):
        return f'“{ev["original_comment"][:90]}”'
    if ev.get("pattern_label"):
        return ev["pattern_label"][:110]
    if ev.get("counts"):
        c = ev["counts"]
        return (f'Approve {c["approve"]} · Disapprove {c["disapprove"]} · '
                f'None {c["none"]}')
    return ""


def render_evidence_item(ev, key_prefix, theme=None):
    """Compact evidence item: ID · type · group · reaction, snippet beneath."""
    with st.container(border=True):
        head = [(ev["evidence_id"], "gray"), (ev["type"], "human")]
        group = ev.get("dim_value") or ev.get("scenario")
        if group:
            head.append((str(group)[:28], "gray"))
        if ev.get("reaction"):
            head.append((ev["reaction"].title(), ev["reaction"]))
        if ev.get("analysis_id"):
            head.append((f'Analysis: {analysis_name(ev["analysis_id"])}'[:38],
                         "human"))
        st.markdown(pills(*head), unsafe_allow_html=True)
        snip = evidence_snippet(ev)
        if snip:
            st.markdown(f'<div class="ces-quote">{snip}</div>',
                        unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            with st.popover("View Source"):
                sub = records_for(ev.get("record_ids", []))
                if ev.get("original_comment"):
                    st.markdown(f'**Complete original comment (never altered):**  \n'
                                f'“{ev["original_comment"]}”')
                for _, r in sub.head(20).iterrows():
                    st.markdown(f'- “{r["comment"]}” — {r["record_id"]} · '
                                f'{r["reaction"]} · {r["source_file"]}')
                if len(sub) > 20:
                    st.caption(f"Showing 20 of {len(sub)} source records.")
        with c2:
            with st.popover("Traceability"):
                prov = provenance_from_records(ev.get("record_ids", []))
                chain = ("PROJECT\n  ↓\nENGAGEMENT ACTIVITY\n  ↓\nDATASET\n  ↓\n"
                         "RAW RECORD"
                         + ("\n  ↓\nANALYSIS" if ev.get("analysis_id") else "")
                         + "\n  ↓\nEVIDENCE ITEM"
                         + ("\n  ↓\nTHEME" if ev.get("theme_id") else ""))
                st.markdown(f'<div class="ces-chain">{chain}</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="ces-meta">'
                    f'<b>Project:</b> {st.session_state.project["project_id"]} — '
                    f'{st.session_state.project["metadata"]["project_name"]}<br>'
                    f'<b>Activity:</b> '
                    f'{", ".join(activity_name(a) for a in prov["activity_ids"]) or "—"}<br>'
                    f'<b>Dataset(s):</b> {", ".join(prov["dataset_ids"]) or "—"}<br>'
                    f'<b>Source file(s):</b> {", ".join(prov["source_files"]) or "—"}<br>'
                    f'<b>Analysis:</b> '
                    f'{analysis_name(ev["analysis_id"]) if ev.get("analysis_id") else "—"}<br>'
                    f'<b>Record IDs:</b> {len(ev.get("record_ids", []))} · '
                    f'<b>Response IDs:</b> {len(prov["response_ids"])}<br>'
                    f'<b>Status:</b> {ev["status"]} · <b>Created:</b> {ev["created"]}'
                    '</div>', unsafe_allow_html=True)
        with c3:
            if st.button("Add to Decision", key=f"{key_prefix}-dec-{ev['evidence_id']}"):
                staged = st.session_state.setdefault("decision_staged_evidence", [])
                if ev["evidence_id"] not in staged:
                    staged.append(ev["evidence_id"])
                st.toast(f'{ev["evidence_id"]} staged for the next decision '
                         '(see Decision Trails)')
        with c4:
            if theme is not None:
                if st.button("Remove from Theme",
                             key=f"{key_prefix}-rm-{ev['evidence_id']}"):
                    ev["theme_id"] = None
                    st.rerun()


def render_theme_expanded(theme):
    """Expanded theme: interpretation, tags, reactions, provenance, evidence."""
    st.markdown(
        f'<div class="ces-meta" style="margin-bottom:8px;">'
        f'<b>Analysis of Origin:</b> '
        f'{analysis_name(theme["analysis_id"]) if theme.get("analysis_id") else "— (created outside an analysis)"} · '
        f'<b>Engagement Activity:</b> '
        f'{", ".join(activity_name(a) for a in theme.get("activity_ids", [])) or "—"} · '
        f'<b>Datasets Represented:</b> '
        f'{", ".join(theme.get("dim_values", [])) or ", ".join(theme.get("dataset_ids", [])) or "—"}'
        '</div>', unsafe_allow_html=True)
    st.markdown("**Theme Interpretation**")
    st.markdown(f'<div class="ces-note-human"><b>Human interpretation:</b> '
                f'{theme["interpretation"] or "—"}</div>', unsafe_allow_html=True)
    if theme.get("ai_original_name"):
        st.markdown(f'<div class="ces-note-ai"><b>Original AI interpretation '
                    f'(preserved):</b> “{theme["ai_original_name"]}” — '
                    f'{theme["ai_original_summary"]}</div>', unsafe_allow_html=True)
    if theme.get("tags"):
        st.markdown("**Tags** " +
                    "".join(pill(t, "human") for t in theme["tags"]),
                    unsafe_allow_html=True)

    cts = theme["counts"]
    total = max(1, sum(cts.values()))
    st.markdown("**Reaction Distribution**")
    rc = st.columns(3)
    rc[0].metric("Approve", f'{cts["approve"]} ({cts["approve"]/total:.0%})')
    rc[1].metric("Disapprove", f'{cts["disapprove"]} ({cts["disapprove"]/total:.0%})')
    rc[2].metric("None", f'{cts["none"]} ({cts["none"]/total:.0%})')

    st.markdown(
        "**Datasets Represented** " +
        pills(*[(s, "gray") for s in theme.get("dim_values", [])]) + "<br>" +
        "**Engagement Activity** " +
        pills(*[(activity_name(a), "gray")
                for a in theme.get("activity_ids", [])]) + "<br>" +
        "**Analysis of Origin** " +
        pills((analysis_name(theme["analysis_id"]), "human")
              if theme.get("analysis_id") else ("—", "gray")),
        unsafe_allow_html=True)

    if theme.get("constraints"):
        st.markdown("**Relevant Constraints**")
        for cid in theme["constraints"]:
            con = next((c for c in st.session_state.constraints
                        if c["id"] == cid), None)
            if con:
                st.markdown(pills(("Human Confirmed", "validated")) +
                            f' {con["id"]} — {con["name"]}',
                            unsafe_allow_html=True)

    tb1, tb2 = st.columns(2)
    if tb1.button("Add Entire Theme to Decision",
                  key=f"themedec-{theme['theme_id']}", type="primary"):
        if theme["theme_id"] not in st.session_state.decision_staged_themes:
            st.session_state.decision_staged_themes.append(theme["theme_id"])
        st.toast(f'Theme “{theme["name"]}” staged for the next decision — '
                 'underlying evidence stays linked')
    if theme.get("analysis_id") and get_analysis(theme["analysis_id"]):
        if tb2.button("View Theme Map", key=f"themetmap-{theme['theme_id']}"):
            an = get_analysis(theme["analysis_id"])
            if "theme_map" not in an["enabled_modules"]:
                an["enabled_modules"].append("theme_map")
            st.session_state.active_analysis_id = theme["analysis_id"]
            st.session_state.theme_map_focus = theme["theme_id"]
            # applied before the nav widget instantiates on the next run
            st.session_state.nav_target = "03 Insights Playground"
            st.rerun()

    st.markdown("---")
    st.markdown("**Evidence Within This Theme**")
    theme_ev = [e for e in st.session_state.evidence
                if e.get("theme_id") == theme["theme_id"]]
    quant_ev = [e for e in theme_ev
                if e["type"] in ("Quantitative Pattern", "Cross-Group Pattern",
                                 "Cross-Scenario Pattern")]
    key_ev = [e for e in theme_ev
              if e["type"] in ("Highlighted Quote", "Direct Comment")]
    other_ev = [e for e in theme_ev if e not in quant_ev and e not in key_ev]

    st.markdown("**Key Evidence**")
    if key_ev:
        for ev in key_ev:
            render_evidence_item(ev, f"th-{theme['theme_id']}", theme=theme)
    else:
        st.caption("No saved quotes or comments assigned yet — save them from the "
                   "Insights Playground and assign to this theme.")

    st.markdown(f"**Supporting Comments** ({len(theme['record_ids'])} records)")
    sub = records_for(theme["record_ids"])
    support = sub[~sub["record_id"].isin(theme.get("counter_ids", []))]
    for _, r in support.head(5).iterrows():
        comment_card(r, f"thsup-{theme['theme_id']}", show_actions=False)
    if len(support) > 5:
        with st.expander(f"All {len(support)} supporting comments"):
            for _, r in support.iterrows():
                comment_card(r, f"thsupall-{theme['theme_id']}",
                             show_actions=False)

    if theme.get("counter_ids"):
        st.markdown(f'<div class="ces-note-warn"><b>Counter-Evidence '
                    f'({len(theme["counter_ids"])}):</b> comments within this theme '
                    'expressing a competing position. Preserved deliberately — a '
                    'recurring theme is not consensus.</div>',
                    unsafe_allow_html=True)
        for rid in theme["counter_ids"][:5]:
            r = get_record(rid)
            if r is not None:
                comment_card(r, f"thctr-{theme['theme_id']}", show_actions=False)

    st.markdown("**Quantitative Patterns**")
    st.markdown(f'<div class="ces-meta">Calculated from this theme\'s records: '
                f'Approve {cts["approve"]} · Disapprove {cts["disapprove"]} · '
                f'None {cts["none"]} across {len(theme["record_ids"])} comments / '
                f'{theme["n_respondents"]} unique response IDs.</div>',
                unsafe_allow_html=True)
    for ev in quant_ev:
        render_evidence_item(ev, f"thq-{theme['theme_id']}", theme=theme)

    if other_ev:
        st.markdown("**Other Related Evidence**")
        for ev in other_ev:
            render_evidence_item(ev, f"tho-{theme['theme_id']}", theme=theme)


def page_libraries():
    st.title("Libraries")
    tab_ev, tab_con = st.tabs(["Evidence Library", "Constraints Library"])

    # ---------------- EVIDENCE LIBRARY (theme-centric) ----------------
    with tab_ev:
        st.markdown(f'<p style="color:{C["text2"]};">Explore validated themes and '
                    'the source material that supports them.</p>',
                    unsafe_allow_html=True)
        themes = st.session_state.themes
        evidence = st.session_state.evidence
        all_theme_records = {rid for t in themes for rid in t["record_ids"]}
        acts = sorted({a for t in themes for a in t.get("activity_ids", [])}
                      | {a for e in evidence for a in e.get("activity_ids", [])})
        dsets = sorted({d for t in themes for d in t.get("dataset_ids", [])}
                       | {d for e in evidence for d in e.get("dataset_ids", [])})

        m = st.columns(4)
        m[0].metric("Validated Themes", len(themes))
        m[1].metric("Total Evidence Items", len(evidence))
        m[2].metric("Engagement Activities", len(acts))
        m[3].metric("Datasets Represented", len(dsets))

        # ---------- filters: themes first, then evidence inside ----------
        an_ids = sorted({t.get("analysis_id") for t in themes
                         if t.get("analysis_id")}
                        | {e.get("analysis_id") for e in evidence
                           if e.get("analysis_id")})
        fc1, fc2, fc3, fc4, fc5, fc6, fc7 = st.columns(
            [1.2, 1.2, 1, 1, 1, 1, 1.3])
        lf_an = fc1.selectbox("Analysis", ["All"] + an_ids,
                              format_func=lambda a: a if a == "All"
                              else f"{a} · {analysis_name(a)}")
        lf_act = fc2.selectbox("Engagement Activity", ["All"] + acts,
                               format_func=lambda a: a if a == "All"
                               else activity_name(a))
        lf_dim = fc3.selectbox(
            "Dataset Value",
            ["All"] + sorted({s for t in themes for s in t.get("dim_values", [])}))
        lf_status = fc4.selectbox("Validation Status", ["All", "Human Validated"])
        lf_tag = fc5.selectbox("Tag", ["All"] + all_known_tags())
        lf_reac = fc6.selectbox("Reaction",
                                ["All", "approve", "disapprove", "none"])
        lf_q = fc7.text_input("Search Themes", placeholder="Search themes…")

        def theme_matches(t):
            if lf_an != "All" and t.get("analysis_id") != lf_an:
                return False
            if lf_act != "All" and lf_act not in t.get("activity_ids", []):
                return False
            if lf_dim != "All" and lf_dim not in t.get("dim_values", []):
                return False
            if lf_status != "All" and t["status"].title() != lf_status:
                return False
            if lf_tag != "All" and lf_tag not in t.get("tags", []):
                return False
            if lf_reac != "All" and t["counts"].get(lf_reac, 0) == 0:
                return False
            if lf_q.strip() and lf_q.strip().lower() not in \
                    (t["name"] + " " + t["interpretation"]).lower():
                return False
            return True

        shown = [t for t in themes if theme_matches(t)]
        if not themes:
            st.caption("No themes yet. Validate an AI cluster or create a human "
                       "theme in the Insights Playground.")
        elif not shown:
            st.caption("No themes match the current filters.")

        for theme in shown:
            n_ev = len([e for e in evidence
                        if e.get("theme_id") == theme["theme_id"]])
            header = (f'{theme["name"]} — {len(theme["record_ids"])} records · '
                      f'{n_ev} saved evidence items · '
                      f'{len(theme.get("dataset_ids", []))} datasets')
            with st.expander(header):
                st.markdown(
                    pills((theme["theme_id"], "gray"),
                          ("Human Validated", "validated"),
                          *([(f'Analysis: {analysis_name(theme["analysis_id"])}',
                              "human")] if theme.get("analysis_id") else []),
                          *[(activity_name(a), "gray")
                            for a in theme.get("activity_ids", [])],
                          *[(s, "gray") for s in theme.get("dim_values", [])]),
                    unsafe_allow_html=True)
                if theme["interpretation"]:
                    st.markdown(f'<div class="ces-meta">“'
                                f'{theme["interpretation"][:140]}”</div>',
                                unsafe_allow_html=True)
                render_theme_expanded(theme)

        # ---------- UNASSIGNED EVIDENCE ----------
        unassigned = [e for e in evidence if not e.get("theme_id")]
        if unassigned:
            st.markdown("---")
            st.markdown(f'<div class="ces-note-yellow"><b>Unassigned Evidence '
                        f'({len(unassigned)})</b> — saved evidence not yet part of '
                        'a theme. Assign it so the library stays organized around '
                        'themes.</div>', unsafe_allow_html=True)
            theme_opts = {t["theme_id"]: t["name"] for t in themes}
            for ev in unassigned:
                render_evidence_item(ev, "un")
                a1, a2 = st.columns(2)
                with a1:
                    if theme_opts:
                        pick = st.selectbox(
                            "Assign to Existing Theme",
                            ["—"] + list(theme_opts),
                            format_func=lambda k: theme_opts.get(k, k),
                            key=f"assign-{ev['evidence_id']}")
                        if pick != "—":
                            ev["theme_id"] = pick
                            st.rerun()
                with a2:
                    with st.popover("Create New Theme"):
                        tn = st.text_input("Theme name (short)",
                                           key=f"nt-{ev['evidence_id']}")
                        ti = st.text_area("Interpretation",
                                          key=f"nti-{ev['evidence_id']}")
                        if st.button("Create theme from this evidence",
                                     key=f"ntb-{ev['evidence_id']}"):
                            if tn.strip():
                                prov = provenance_from_records(
                                    ev.get("record_ids", []))
                                sub = records_for(ev.get("record_ids", []))
                                theme_id = next_id("theme_seq", "TH-")
                                ev_an = get_analysis(ev.get("analysis_id"))
                                st.session_state.themes.append({
                                    "theme_id": theme_id, "origin": "human",
                                    "analysis_id": ev.get("analysis_id"),
                                    "dimension": (
                                        ev_an["comparison_dimension"]["name"]
                                        if ev_an else None),
                                    "ai_original_name": None,
                                    "ai_original_summary": None,
                                    "ai_source": None, "name": tn.strip(),
                                    "interpretation": ti.strip(),
                                    "record_ids": ev.get("record_ids", []),
                                    "excluded_record_ids": [],
                                    "counts": reaction_counts(sub) if len(sub)
                                    else {"approve": 0, "disapprove": 0,
                                          "none": 0},
                                    "n_respondents":
                                        int(sub["response_id"].nunique())
                                        if len(sub) else 0,
                                    "tags": ev.get("tags", []), "notes": "",
                                    "counter_ids": [], "constraints": [],
                                    "validated":
                                        datetime.date.today().isoformat(),
                                    "status": "HUMAN VALIDATED",
                                    "cluster_key": None, **prov})
                                ev["theme_id"] = theme_id
                                st.rerun()

    # ---------------- CONSTRAINTS LIBRARY ----------------
    with tab_con:
        st.markdown(f'<p style="color:{C["text2"]};">Project-level conditions that '
                    'shape what is possible. Linkable to themes, evidence, and '
                    'decisions.</p>', unsafe_allow_html=True)
        if not st.session_state.constraints:
            st.caption("No constraints documented yet — add them in Data + Context.")
        for con in st.session_state.constraints:
            related_themes = [t for t in st.session_state.themes
                              if con["id"] in t.get("constraints", [])]
            related_dec = [d["id"] for d in st.session_state.decisions
                           if con["id"] in d.get("constraints", [])]
            with st.container(border=True):
                st.markdown(pills(
                    (con["id"], "gray"),
                    (con["type"], "conflict" if "Legal" in con["type"] or
                     "Voter" in con["type"] else "human"),
                    (con["status"], "validated")), unsafe_allow_html=True)
                st.markdown(f"**{con['name']}**")
                if con["description"]:
                    st.write(con["description"])
                st.markdown(
                    f'<div class="ces-meta">Source: {con["source"] or "—"} · '
                    f'Phase: {con["phase"] or "—"}<br>'
                    f'Related themes: '
                    f'{", ".join(t["name"] for t in related_themes) or "none"} · '
                    f'Related decisions: {", ".join(related_dec) or "none"}</div>',
                    unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Add to Decision", key=f"condec-{con['id']}"):
                        staged = st.session_state.setdefault(
                            "decision_staged_constraints", [])
                        if con["id"] not in staged:
                            staged.append(con["id"])
                        st.toast(f'{con["id"]} staged for the next decision')
                with b2:
                    with st.popover("Traceability"):
                        chain = ("SOURCE DOCUMENT\n  ↓\nDOCUMENTED CONSTRAINT\n"
                                 "  ↓\nHUMAN VALIDATION\n  ↓\nRELEVANT DECISION")
                        st.markdown(f'<div class="ces-chain">{chain}</div>',
                                    unsafe_allow_html=True)
                        st.markdown(f'<div class="ces-meta"><b>Source:</b> '
                                    f'{con["source"] or "—"}</div>',
                                    unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE 05 — DECISION TRAILS
# ----------------------------------------------------------------------------

DECISION_DIAGRAM = (
    "COMMUNITY EVIDENCE ─────┐\n"
    "                        │\n"
    "PROJECT CONSTRAINTS ────┼→ CONSIDERATION → DECISION\n"
    "                        │\n"
    "CONFLICTING INPUT ──────┘"
)


def page_decisions():
    st.title("Decision Trails")
    st.markdown(f'<p style="color:{C["text2"]};">Document how themes, evidence, '
                'constraints, trade-offs, and judgment shaped a planning decision. '
                'Inputs can be pulled from <b>multiple analyses</b> and multiple '
                'engagement activities — each input preserves where it '
                'originated. Inputs are <b>considered</b> — evidence does not '
                'automatically cause a decision.</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="ces-chain">{DECISION_DIAGRAM}</div>',
                unsafe_allow_html=True)

    theme_opts = {t["theme_id"]: f'{t["theme_id"]} · {t["name"]} '
                                 f'({len(t["record_ids"])} records'
                                 + (f' · {analysis_name(t["analysis_id"])}'
                                    if t.get("analysis_id") else "") + ")"
                  for t in st.session_state.themes}
    ev_opts = {e["evidence_id"]: f'{e["evidence_id"]} · {e["type"]} · '
                                 f'{evidence_snippet(e)[:50]}'
               for e in st.session_state.evidence}
    con_opts = {c["id"]: f'{c["id"]} · {c["name"]}'
                for c in st.session_state.constraints}
    staged_t = [t for t in st.session_state.decision_staged_themes
                if t in theme_opts]
    staged_e = [e for e in st.session_state.get("decision_staged_evidence", [])
                if e in ev_opts]
    staged_c = [c for c in st.session_state.get("decision_staged_constraints", [])
                if c in con_opts]
    if staged_t or staged_e or staged_c:
        st.markdown('<div class="ces-note-human">Items staged from the Libraries '
                    'are pre-selected in the form below.</div>',
                    unsafe_allow_html=True)

    with st.expander("＋ New Decision",
                     expanded=(len(st.session_state.decisions) == 0
                               or bool(staged_t or staged_e or staged_c))):
        with st.form("decision-form", clear_on_submit=True):
            d_name = st.text_input("Decision Name")
            d_desc = st.text_area("Decision Description")
            c1, c2 = st.columns(2)
            d_date = c1.text_input("Date", datetime.date.today().isoformat())
            d_maker = c2.text_input("Decision Maker")
            d_alts = st.text_area("Alternatives Considered", height=60)
            d_rationale = st.text_area("Rationale", height=60)
            st.markdown("**Community Evidence**")
            d_themes = st.multiselect("Add Theme (whole theme, evidence stays "
                                      "linked)", list(theme_opts),
                                      default=staged_t,
                                      format_func=lambda k: theme_opts[k])
            d_evidence = st.multiselect("Add Evidence (individual items)",
                                        list(ev_opts), default=staged_e,
                                        format_func=lambda k: ev_opts[k])
            st.markdown("**Project Constraints**")
            d_constraints = st.multiselect("Add Constraint", list(con_opts),
                                           default=staged_c,
                                           format_func=lambda k: con_opts[k])
            st.markdown("**Trade-offs / Conflicts**")
            d_conflict_ev = st.multiselect(
                "Conflicting or trade-off input (from Evidence Library)",
                list(ev_opts), format_func=lambda k: ev_opts[k],
                key="dec-conflict")
            d_tradeoffs = st.text_area("Trade-off notes", height=60)
            if st.form_submit_button("Save Decision", type="primary"):
                if not d_name.strip():
                    st.error("The decision needs a name.")
                else:
                    st.session_state.decision_seq += 1
                    st.session_state.decisions.append({
                        "id": f"DEC-{st.session_state.decision_seq:03d}",
                        "name": d_name.strip(), "description": d_desc.strip(),
                        "date": d_date, "maker": d_maker.strip(),
                        "alternatives": d_alts.strip(),
                        "rationale": d_rationale.strip(),
                        "themes": d_themes, "evidence": d_evidence,
                        "constraints": d_constraints,
                        "conflicts": d_conflict_ev,
                        "tradeoffs": d_tradeoffs.strip()})
                    st.session_state.decision_staged_themes = []
                    st.session_state.decision_staged_evidence = []
                    st.session_state.decision_staged_constraints = []
                    st.rerun()

    for dec in st.session_state.decisions:
        with st.container(border=True):
            st.markdown(pills((dec["id"], "gray"), ("DECISION", "human")),
                        unsafe_allow_html=True)
            st.markdown(f"### {dec['name']}")
            if dec["description"]:
                st.write(dec["description"])
            st.markdown(f'<div class="ces-meta">Date: {dec["date"]} · '
                        f'Decision maker: {dec["maker"] or "—"}</div>',
                        unsafe_allow_html=True)
            col_e, col_c, col_x = st.columns(3)
            with col_e:
                st.markdown("**Community Evidence**")
                for tid in dec.get("themes", []):
                    th = get_theme(tid)
                    if th:
                        origin = (f' · from {analysis_name(th["analysis_id"])}'
                                  if th.get("analysis_id") else "")
                        st.markdown(pills(("THEME", "validated"),
                                          (th["name"], "gray")) +
                                    f'<span class="ces-meta"> '
                                    f'{len(th["record_ids"])} records'
                                    f'{origin}</span>',
                                    unsafe_allow_html=True)
                        with st.expander("View Evidence Used"):
                            st.markdown(f'<div class="ces-meta">'
                                        f'{th["interpretation"] or ""}</div>',
                                        unsafe_allow_html=True)
                            for _, r in records_for(
                                    th["record_ids"]).head(10).iterrows():
                                st.markdown(f'- “{r["comment"][:90]}” '
                                            f'({r["record_id"]})')
                            linked = [e for e in st.session_state.evidence
                                      if e.get("theme_id") == tid]
                            for e in linked:
                                st.markdown(f'- {e["evidence_id"]} · {e["type"]}')
                for eid in dec.get("evidence", []):
                    st.markdown(pill(eid, "validated"), unsafe_allow_html=True)
                if not dec.get("themes") and not dec.get("evidence"):
                    st.caption("none attached")
            with col_c:
                st.markdown("**Project Constraints**")
                for cid in dec.get("constraints", []):
                    con = next((c for c in st.session_state.constraints
                                if c["id"] == cid), None)
                    st.markdown(pill(cid, "review") +
                                (f'<span class="ces-meta"> {con["name"]}</span>'
                                 if con else ""), unsafe_allow_html=True)
                if not dec.get("constraints"):
                    st.caption("none attached")
            with col_x:
                st.markdown("**Trade-offs / Conflicts**")
                for eid in dec.get("conflicts", []):
                    st.markdown(pill(eid, "conflict"), unsafe_allow_html=True)
                if dec.get("tradeoffs"):
                    st.caption(dec["tradeoffs"])
                if not dec.get("conflicts") and not dec.get("tradeoffs"):
                    st.caption("none documented")
            if dec.get("alternatives"):
                st.markdown(f'<div class="ces-meta"><b>Alternatives considered:'
                            f'</b> {dec["alternatives"]}</div>',
                            unsafe_allow_html=True)
            if dec.get("rationale"):
                st.markdown(f'<div class="ces-note-human"><b>Rationale:</b> '
                            f'{dec["rationale"]}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------------

def main():
    init_state()
    if "nav_target" in st.session_state:
        st.session_state.nav = st.session_state.pop("nav_target")
    proj = st.session_state.project
    with st.sidebar:
        st.markdown("## Civic Evidence Studio")
        st.caption(f'{proj["metadata"]["project_name"]} · '
                   f'{proj["metadata"]["project_phase"]}')
        page = st.radio("Navigate", ["01 Data + Context", "02 Analysis Setup",
                                     "03 Insights Playground", "04 Libraries",
                                     "05 Decision Trails"],
                        key="nav", label_visibility="collapsed")
        st.divider()
        n_act = len(all_activities())
        n_ds = sum(len(a["datasets"]) for a in all_activities())
        processed = sum(1 for a in all_activities() if a["combined"] is not None)
        active = active_analysis()
        st.caption(
            f"Activities: {n_act} ({processed} processed)\n\n"
            f"Datasets: {n_ds}\n\n"
            f"Analyses: {len(st.session_state.analyses)}"
            + (f" (active: {active['analysis_name']})" if active else "")
            + "\n\n"
            f"Themes: {len(st.session_state.themes)}\n\n"
            f"Evidence items: {len(st.session_state.evidence)}\n\n"
            f"Constraints: {len(st.session_state.constraints)}\n\n"
            f"Decisions: {len(st.session_state.decisions)}")
        provider, _ = llm_provider()
        st.markdown(pill("AI: " + (provider if provider else "not configured"),
                         "ai" if provider else "review"), unsafe_allow_html=True)
        st.caption("PROJECT → ACTIVITY → DATASET → RECORD → ANALYSIS → "
                   "AI-SUGGESTED PATTERN → HUMAN THEME → EVIDENCE → DECISION. "
                   "Provenance is preserved at every level.")

    if page == "01 Data + Context":
        page_data_context()
    elif page == "02 Analysis Setup":
        page_analysis_setup()
    elif page == "03 Insights Playground":
        page_insights()
    elif page == "04 Libraries":
        page_libraries()
    else:
        page_decisions()


main()
