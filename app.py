"""
CIVIC EVIDENCE STUDIO — functional Streamlit prototype (v2)
===========================================================
Information hierarchy:

  PROJECT
    └── ENGAGEMENT ACTIVITIES        (one activity can contain MANY files)
          └── DATASETS / FILES
                └── RAW RECORDS
                      → THEMES → EVIDENCE  (+ PROJECT CONSTRAINTS) → DECISION TRAILS

Provenance is preserved at every level: each processed record carries
project_id, activity_id, dataset_id, source_file, scenario, topic,
record_id, response_id, the unaltered original comment, and reaction.

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


def init_state():
    ss = st.session_state
    if "project" not in ss:
        ss.project = {
            "project_id": "PROJ-001",
            "metadata": dict(DEFAULT_PROJECT_METADATA),
            "engagement_activities": [
                {"activity_id": "ENG-001",
                 "metadata": dict(DEFAULT_COMAP_ACTIVITY),
                 "datasets": [],          # dataset dicts (see add_dataset)
                 "combined": None}        # processed activity dataframe
            ],
        }
    ss.setdefault("activity_seq", 1)
    ss.setdefault("dataset_seq", 0)
    ss.setdefault("constraints", [])          # PROJECT-level constraints
    ss.setdefault("constraint_seq", 0)
    ss.setdefault("clusters", {})             # cluster_key -> cluster dict
    ss.setdefault("cluster_run_done", False)
    ss.setdefault("cluster_scope_desc", "")
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


def guess_scenario(filename):
    m = re.search(r"scenario\s*_?\s*(\d)", filename, re.IGNORECASE)
    return f"Scenario {m.group(1)}" if m else ""


def guess_topic(filename):
    for topic in ("Housing", "Mobility", "Ecology", "Water", "Revenue", "Connectivity"):
        if topic.lower() in filename.lower():
            return topic
    return ""


def standardize_dataset(raw_df, dataset, activity):
    """Normalize an uploaded file into standard records with full provenance.
    Returns (df, problems). Original comments are never altered."""
    problems = []
    df = raw_df.copy()
    df.columns = [re.sub(r"[\s_]+", "", str(c).strip().lower()) for c in df.columns]
    colmap = {}
    for c in df.columns:
        if c in ("comment", "comments", "commenttext", "text"):
            colmap[c] = "comment"
        elif c in ("reaction", "reactions", "thumb", "thumbs", "vote"):
            colmap[c] = "reaction"
        elif c in ("responseid", "respid", "responseld", "response", "participantid"):
            colmap[c] = "response_id"
    df = df.rename(columns=colmap)
    for req, label in [("comment", "Comment"), ("reaction", "Reaction"),
                       ("response_id", "Response ID")]:
        if req not in df.columns:
            problems.append(
                f"**{dataset['source_file']}** is missing a required column: "
                f"**{label}**. Columns found: {', '.join(df.columns)}. "
                "The app will not fabricate this field.")
    if problems:
        return None, problems

    df["comment"] = df["comment"].astype(str)
    df["response_id"] = df["response_id"].astype(str).str.strip()
    df["reaction_original"] = df["reaction"]
    df["reaction"] = df["reaction"].apply(_norm_reaction)
    df = df.reset_index(drop=True)

    num = dataset["dataset_id"].split("-")[-1]
    df["record_id"] = [f"D{num}-{i + 1:05d}" for i in range(len(df))]
    df["project_id"] = st.session_state.project["project_id"]
    df["activity_id"] = activity["activity_id"]
    df["dataset_id"] = dataset["dataset_id"]
    df["source_file"] = dataset["source_file"]
    df["scenario"] = dataset["scenario"] or "(unspecified)"
    df["topic"] = dataset["topic"] or ""

    cols = ["project_id", "activity_id", "dataset_id", "source_file", "scenario",
            "topic", "record_id", "response_id", "comment", "reaction",
            "reaction_original"]
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


def llm_interpret_cluster(sample_comments, keywords, scope_label):
    provider, key = llm_provider()
    if provider is None:
        return None
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
    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model="claude-sonnet-4-5", max_tokens=400,
                messages=[{"role": "user", "content": prompt}])
            text = msg.content[0].text
        else:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini", max_tokens=400,
                messages=[{"role": "user", "content": prompt}])
            text = resp.choices[0].message.content
        data = _extract_json(text)
        if data and data.get("name"):
            return {"name": str(data.get("name", "")).strip(),
                    "summary": str(data.get("summary", "")).strip(),
                    "tags": [str(t).strip() for t in (data.get("tags") or [])][:5]}
    except Exception as e:
        st.session_state.setdefault("llm_errors", []).append(str(e))
    return None


# ----------------------------------------------------------------------------
# THEMATIC CLUSTERING (local, transparent: TF-IDF + KMeans)
# ----------------------------------------------------------------------------

def cluster_one_group(sdf):
    """Cluster comments for one group (usually one scenario). Calculated fields
    only — AI interpretation attached separately."""
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
        counts = reaction_counts(sub)
        if counts["approve"] >= counts["disapprove"]:
            majority, minority = "approve", "disapprove"
        else:
            majority, minority = "disapprove", "approve"
        clusters.append({
            "scenario": sub["scenario"].iloc[0],
            "record_ids": sub["record_id"].tolist(),
            "dataset_ids": sorted(sub["dataset_id"].unique().tolist()),
            "activity_ids": sorted(sub["activity_id"].unique().tolist()),
            "source_files": sorted(sub["source_file"].unique().tolist()),
            "n_comments": int(len(sub)),
            "n_respondents": int(sub["response_id"].nunique()),
            "counts": counts, "majority": majority,
            "counter_ids": sub[sub["reaction"] == minority]["record_id"].tolist(),
            "keywords": top_terms, "rep_ids": rep_ids,
        })
    clusters.sort(key=lambda c: -c["n_comments"])
    return clusters


def run_thematic_analysis(scope_df, scope_desc):
    """Cluster the scoped records, grouped by scenario; attach AI interpretation.
    Dataset/activity provenance is preserved inside every cluster."""
    provider, _ = llm_provider()
    st.session_state.clusters = {}
    groups = sorted(scope_df["scenario"].unique())
    for scen in groups:
        sdf = scope_df[scope_df["scenario"] == scen]
        found = cluster_one_group(sdf)
        for j, cl in enumerate(found):
            key = f"{re.sub(r'[^A-Za-z0-9]', '', scen)}-C{j + 1}"
            cl["key"] = key
            cl["status"] = "ai-suggested"
            cl["constraint_links"] = {}
            rep_comments = records_for(cl["rep_ids"])["comment"].tolist()
            ai = llm_interpret_cluster(rep_comments, cl["keywords"],
                                       f"{scope_desc}, {scen}") if provider else None
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
    st.session_state.cluster_run_done = True
    st.session_state.cluster_scope_desc = scope_desc


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
    """Create an evidence item. theme_id may be None → Unassigned Evidence."""
    item["evidence_id"] = next_id("evidence_seq", "EV-", width=4)
    item["created"] = datetime.date.today().isoformat()
    item.setdefault("theme_id", None)
    st.session_state.evidence.append(item)
    return item["evidence_id"]


def provenance_from_records(record_ids):
    sub = records_for(record_ids)
    if sub.empty:
        return {"activity_ids": [], "dataset_ids": [], "scenarios": [],
                "source_files": [], "response_ids": []}
    return {
        "activity_ids": sorted(sub["activity_id"].unique().tolist()),
        "dataset_ids": sorted(sub["dataset_id"].unique().tolist()),
        "scenarios": sorted(sub["scenario"].unique().tolist()),
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
                  (row["scenario"], "gray")) + tag_html
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
                        "scenario": row["scenario"],
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
                            "scenario": row["scenario"],
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
    n = len(ds["df"]) if ds["df"] is not None else 0
    label = f'{ds["dataset_name"] or ds["source_file"]} — {n} comments' \
            + ("" if ds["include"] else "  (excluded from analysis)")
    with st.expander(label):
        st.markdown(
            pills((ds["dataset_id"], "gray"),
                  (ds["scenario"] or "no scenario", "gray"),
                  (ds["topic"] or "no topic", "gray"),
                  ("Included" if ds["include"] else "Excluded",
                   "validated" if ds["include"] else "review")),
            unsafe_allow_html=True)
        st.markdown(f'<div class="ces-meta">Source: {ds["source_file"]} · '
                    f'{n} comments · '
                    f'{ds["df"]["response_id"].nunique() if ds["df"] is not None else 0} '
                    f'unique response IDs</div>', unsafe_allow_html=True)
        with st.form(f"dsmeta-{ds['dataset_id']}"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Dataset Name", ds["dataset_name"],
                                     placeholder="e.g. Housing — Scenario 1")
                scenario = st.text_input("Scenario / Subgroup", ds["scenario"],
                                         placeholder="e.g. Scenario 1")
            with c2:
                topic = st.text_input("Topic / Category", ds["topic"],
                                      placeholder="e.g. Housing")
                include = st.checkbox("Include in analysis", value=ds["include"])
            desc = st.text_area("Dataset Description (optional)", ds["description"],
                                height=68)
            notes = st.text_area("Dataset Notes (optional)", ds["notes"], height=68)
            if st.form_submit_button("Save Dataset Metadata", type="primary"):
                ds.update({"dataset_name": name.strip(), "scenario": scenario.strip(),
                           "topic": topic.strip(), "description": desc.strip(),
                           "notes": notes.strip(), "include": include})
                activity["combined"] = None  # metadata changed → reprocess
                st.rerun()
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
            if st.form_submit_button("Save Activity Metadata", type="primary"):
                activity["metadata"] = {
                    "activity_name": a_name, "engagement_method": a_method,
                    "activity_date": a_date, "location": a_loc,
                    "stakeholder_groups": a_stake, "participant_count": a_count,
                    "facilitator": a_fac, "purpose": a_purpose, "notes": a_notes}
                st.rerun()

        st.markdown("---")
        st.markdown("**Datasets / Files** — one activity can contain many files. "
                    "File metadata only describes which subset of the activity the "
                    "dataset represents.")

        ups = st.file_uploader(
            "Add file(s) to this activity (XLSX with columns: comment, reaction, "
            "responseId)", type=["xlsx"], accept_multiple_files=True,
            key=f"up-{activity['activity_id']}")
        known_files = {d["source_file"] for d in activity["datasets"]}
        added_new = False
        for up in ups or []:
            if up.name in known_files:
                continue
            ds_id = next_id("dataset_seq", "DATA-")
            ds = {"dataset_id": ds_id,
                  "dataset_name": (f'{guess_topic(up.name) or "Dataset"} — '
                                   f'{guess_scenario(up.name)}').strip(" —"),
                  "source_file": up.name,
                  "scenario": guess_scenario(up.name),
                  "topic": guess_topic(up.name),
                  "description": "", "notes": "", "include": True,
                  "df": None, "problems": []}
            try:
                raw_df = pd.read_excel(up)
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
                    f["scenario"] = d["scenario"] or "(unspecified)"
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
            per = df.groupby(["dataset_id", "source_file", "scenario"]).agg(
                comments=("record_id", "count"),
                unique_respondents=("response_id", "nunique")).reset_index()
            st.dataframe(per, width="stretch", hide_index=True)
            st.caption("Every row retains project_id, activity_id, dataset_id, "
                       "source_file, scenario, topic, record_id, and response_id — "
                       "relationships are never flattened away.")


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
# PAGE 02 — INSIGHTS PLAYGROUND
# ----------------------------------------------------------------------------

def reaction_chart(df, group_col="scenario"):
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
        xaxis=dict(linecolor=C["border"]))
    return fig


def scenario_pct_table(df):
    rows = []
    for s in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == s]
        cts = reaction_counts(sdf)
        n = max(1, len(sdf))
        rows.append({"Scenario": s, "Comments": len(sdf),
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
        st.markdown(pills(src_pill, (cluster["scenario"], "gray"), *status_pill),
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
                          if k != key and c["status"] == "ai-suggested"]
                if others:
                    target = st.selectbox(
                        "Merge into", others,
                        format_func=lambda k:
                        f'{st.session_state.clusters[k]["ai"]["name"]} '
                        f'({st.session_state.clusters[k]["scenario"]})',
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
        theme = {
            "theme_id": theme_id, "origin": "ai-validated",
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
                    st.session_state.themes.append({
                        "theme_id": theme_id, "origin": "human",
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


def cross_scenario_patterns():
    clusters = [c for c in st.session_state.clusters.values()
                if c["status"] in ("ai-suggested", "validated")]
    by_scen = {}
    for c in clusters:
        by_scen.setdefault(c["scenario"], []).append(c)
    scens = sorted(by_scen)
    patterns, seen = [], set()
    for i, s1 in enumerate(scens):
        for c1 in by_scen[s1]:
            group = [c1]
            kws1 = set(c1["keywords"])
            for s2 in scens[i + 1:]:
                best, best_j = None, 0.0
                for c2 in by_scen[s2]:
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
                n_scen = len({c["scenario"] for c in group})
                if n_scen == len(scens) and len(scens) >= 3:
                    rel = "Appears Across All Scenarios"
                else:
                    dis = {c["scenario"]: (c["counts"]["disapprove"] /
                                           max(1, c["n_comments"])) for c in group}
                    strongest = max(dis, key=dis.get)
                    rel = (f"Stronger in {strongest}"
                           if max(dis.values()) - min(dis.values()) > 0.2
                           else "Shared Across Scenarios")
                patterns.append({"key": "|".join(gkey), "relationship": rel,
                                 "clusters": group,
                                 "shared_keywords": sorted(set.intersection(
                                     *[set(c["keywords"]) for c in group]))})
    for c in clusters:
        if not any(c in p["clusters"] for p in patterns):
            patterns.append({"key": c["key"],
                             "relationship": "Mostly Scenario-Specific",
                             "clusters": [c],
                             "shared_keywords": c["keywords"][:4]})
    return patterns


def page_insights():
    st.title("Insights Playground")
    st.markdown(
        pills(("AI-assisted analysis", "ai")) +
        f'<span style="color:{C["text2"]};font-size:13px;"> AI suggestions are '
        'starting points for human interpretation.</span>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{C["text2"]};margin-top:-2px;">Explore the '
                'records, compare datasets, and develop themes — within one dataset, '
                'across selected datasets, or across the whole engagement '
                'activity.</p>', unsafe_allow_html=True)

    frames = analysis_frames()
    if not frames:
        st.markdown('<div class="ces-note-human">No processed activity dataset yet. '
                    'Go to <b>01 Data + Context</b>, upload files into the '
                    'engagement activity, and press <b>Process Activity '
                    'Datasets</b>.</div>', unsafe_allow_html=True)
        return

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

    # ---------- HIERARCHY-AWARE FILTER BAR ----------
    act_ids = list(frames.keys())
    f1, f2, f3, f4, f5, f6 = st.columns([1.4, 1.4, 1, 1, 1.2, 1.4])
    f_act = f1.selectbox("Engagement Activity", act_ids,
                         format_func=activity_name)
    adf = frames[f_act]
    activity = get_activity(f_act)
    ds_opts = {d["dataset_id"]: (d["dataset_name"] or d["source_file"])
               for d in activity["datasets"] if d["include"] and d["df"] is not None}
    f_ds = f2.selectbox("Dataset", ["All"] + list(ds_opts),
                        format_func=lambda k: ds_opts.get(k, k))
    f_scen = f3.selectbox("Scenario", ["All"] + sorted(adf["scenario"].unique()))
    f_reac = f4.selectbox("Reaction", ["All", "approve", "disapprove", "none"])
    theme_opts = (["All"] +
                  [f'{c["key"]} · {c["ai"]["name"]}'
                   for c in st.session_state.clusters.values()
                   if c["status"] in ("ai-suggested", "validated")] +
                  [f"tag:{t}" for t in all_known_tags()])
    f_theme = f5.selectbox("Theme / Tag", theme_opts)
    f_kw = f6.text_input("Keyword", placeholder="Search comments...")

    view = adf
    if f_ds != "All":
        view = view[view["dataset_id"] == f_ds]
    if f_scen != "All":
        view = view[view["scenario"] == f_scen]
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

    st.caption(f"{activity_name(f_act)} · {len(view)} comments · "
               f"{view['response_id'].nunique()} unique response IDs match the "
               f"current filters (activity total: {len(adf)}).")

    tab_over, tab_comments, tab_themes, tab_compare = st.tabs(
        ["Overview", "Comments", "Themes", "Compare"])

    # ---------------- OVERVIEW ----------------
    with tab_over:
        m = st.columns(4)
        m[0].metric("Activity comments", len(adf))
        m[1].metric("Filtered comments", len(view))
        m[2].metric("Datasets", len(ds_opts))
        m[3].metric("Unique response IDs", adf["response_id"].nunique())
        st.subheader("Participant Reaction by Scenario")
        st.caption("The reaction field supplied in the dataset is the authoritative "
                   "reaction variable. This is not AI sentiment.")
        st.plotly_chart(reaction_chart(view if len(view) else adf),
                        width="stretch")
        st.dataframe(scenario_pct_table(view if len(view) else adf),
                     width="stretch", hide_index=True)
        if st.button("Save this reaction breakdown as a Quantitative Pattern"):
            src = view if len(view) else adf
            add_evidence({
                "type": "Quantitative Pattern",
                "record_ids": src["record_id"].tolist(),
                "scenario": ", ".join(sorted(src["scenario"].unique())),
                "reaction": None, "counts": reaction_counts(src),
                "per_scenario": {s: reaction_counts(src[src["scenario"] == s])
                                 for s in sorted(src["scenario"].unique())},
                "original_comment": None, "selected_quote": None,
                "tags": [], "theme_id": None, "status": "Human Selected",
                **provenance_from_records(src["record_id"].tolist())})
            st.toast("Quantitative pattern saved (Unassigned — assign it to a "
                     "theme in the Evidence Library)")

    # ---------------- COMMENTS ----------------
    with tab_comments:
        with st.expander("Bulk tagging (select multiple comments)"):
            label_map = {rid: f'{rid} · {c[:70]}'
                         for rid, c in zip(view["record_id"], view["comment"])}
            sel = st.multiselect("Comments", view["record_id"].tolist(),
                                 format_func=lambda r: label_map.get(r, r))
            bt1, bt2 = st.columns(2)
            btag = bt1.selectbox("Tag", ["—"] + all_known_tags(), key="bulk-tag-sel")
            bnew = bt2.text_input("Or new tag", key="bulk-tag-new")
            if st.button("Apply tag to selected"):
                chosen = bnew.strip() or (btag if btag != "—" else "")
                if chosen and sel:
                    for rid in sel:
                        add_tag(rid, chosen, "human")
                    st.rerun()
        page_size = 15
        n_pages = max(1, (len(view) - 1) // page_size + 1)
        pg = st.number_input("Page", 1, n_pages, 1) if n_pages > 1 else 1
        for _, r in view.iloc[(pg - 1) * page_size: pg * page_size].iterrows():
            comment_card(r, "cx")
            memberships = [c["ai"]["name"] for c in
                           st.session_state.clusters.values()
                           if r["record_id"] in c["record_ids"]
                           and c["status"] in ("ai-suggested", "validated")]
            if memberships:
                st.caption("In themes: " + " · ".join(memberships))

    # ---------------- THEMES ----------------
    with tab_themes:
        provider, _ = llm_provider()
        if provider is None:
            st.markdown('<div class="ces-note-warn">AI theme interpretation '
                        'unavailable. Configure an API key (ANTHROPIC_API_KEY or '
                        'OPENAI_API_KEY) to enable AI-generated theme names and '
                        'summaries. Local clustering still runs; all human '
                        'workflows remain available.</div>', unsafe_allow_html=True)

        st.markdown("**Analyze:**")
        scope = st.radio(
            "Analysis scope",
            ["Current Dataset", "Selected Datasets", "Entire Engagement Activity"],
            horizontal=True, label_visibility="collapsed")
        if scope == "Current Dataset":
            if f_ds == "All":
                st.caption("Select a single dataset in the filter bar above, or "
                           "choose a wider scope.")
                scope_df = adf
                scope_desc = f"{activity_name(f_act)} (all datasets)"
            else:
                scope_df = adf[adf["dataset_id"] == f_ds]
                scope_desc = f"{ds_opts.get(f_ds, f_ds)}"
        elif scope == "Selected Datasets":
            sel_ds = st.multiselect("Datasets to analyze", list(ds_opts),
                                    default=list(ds_opts),
                                    format_func=lambda k: ds_opts[k])
            scope_df = adf[adf["dataset_id"].isin(sel_ds)]
            scope_desc = f"{activity_name(f_act)} — {len(sel_ds)} selected datasets"
        else:
            scope_df = adf
            scope_desc = f"{activity_name(f_act)} (entire activity)"

        c1, c2 = st.columns([1, 3])
        if c1.button("Run Thematic Analysis", type="primary",
                     disabled=(len(scope_df) == 0)):
            with st.spinner("Clustering per scenario (TF-IDF + KMeans)"
                            + (" and asking the LLM to interpret each cluster…"
                               if provider else "…")):
                run_thematic_analysis(scope_df, scope_desc)
            st.rerun()
        c2.caption(f"Scope: {scope_desc} · {len(scope_df)} comments. Comments are "
                   "clustered per scenario; dataset and activity provenance is "
                   "preserved inside every cluster and theme.")

        render_create_human_theme(adf)

        if st.session_state.validating_cluster:
            cl = st.session_state.clusters.get(st.session_state.validating_cluster)
            if cl:
                render_validation_form(cl)
        if st.session_state.cluster_run_done:
            st.caption(f"Showing clusters from: "
                       f"{st.session_state.cluster_scope_desc}")
            for scen in sorted({c["scenario"] for c in
                                st.session_state.clusters.values()}):
                scl = [c for c in st.session_state.clusters.values()
                       if c["scenario"] == scen
                       and c["status"] in ("ai-suggested", "validated")]
                if scl:
                    st.subheader(scen)
                    for cl in scl:
                        render_cluster_card(cl)
        elif not st.session_state.themes:
            st.caption("No thematic analysis yet — press Run Thematic Analysis.")

    # ---------------- COMPARE ----------------
    with tab_compare:
        st.subheader("Participant Reaction — Cross-Dataset Comparison")
        st.plotly_chart(reaction_chart(adf), width="stretch",
                        key="compare-chart")
        st.dataframe(scenario_pct_table(adf), width="stretch",
                     hide_index=True)
        st.subheader("Themes by Scenario")
        if not st.session_state.cluster_run_done:
            st.caption("Run the thematic analysis in the Themes tab first.")
        else:
            scens = sorted({c["scenario"] for c in
                            st.session_state.clusters.values()})
            scen_cols = st.columns(max(1, len(scens)))
            for i, scen in enumerate(scens):
                with scen_cols[i]:
                    st.markdown(f"**{scen}**")
                    for c in [c for c in st.session_state.clusters.values()
                              if c["scenario"] == scen
                              and c["status"] in ("ai-suggested", "validated")]:
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
            st.subheader("Patterns Across Scenarios")
            st.caption("AI-computed proposals from keyword overlap between "
                       "clusters. All linked to the same engagement activity; "
                       "proposals remain proposals until reviewed by a human.")
            for p in cross_scenario_patterns():
                reviewed = st.session_state.cross_reviews.get(p["key"])
                with st.container(border=True):
                    st.markdown(pills(
                        (p["relationship"], "ai"),
                        *([("Reviewed", "validated")] if reviewed else [])),
                        unsafe_allow_html=True)
                    names = " + ".join(f'{c["ai"]["name"]} ({c["scenario"]})'
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
                        if st.button("Save as Cross-Scenario Pattern (evidence)",
                                     key=f"xssave-{p['key']}"):
                            rids = sorted({rid for c in p["clusters"]
                                           for rid in c["record_ids"]})
                            sub = records_for(rids)
                            add_evidence({
                                "type": "Cross-Scenario Pattern",
                                "record_ids": rids,
                                "scenario": ", ".join(
                                    sorted(sub["scenario"].unique())),
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
                            st.toast("Cross-scenario pattern saved (Unassigned)")


# ----------------------------------------------------------------------------
# PAGE 03 — LIBRARIES  (theme-centric Evidence Library)
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
    """Compact evidence item: ID · type · scenario · reaction, snippet beneath."""
    with st.container(border=True):
        head = [(ev["evidence_id"], "gray"), (ev["type"], "human")]
        if ev.get("scenario"):
            head.append((str(ev["scenario"])[:28], "gray"))
        if ev.get("reaction"):
            head.append((ev["reaction"].title(), ev["reaction"]))
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
                         "RAW RECORD\n  ↓\nEVIDENCE ITEM"
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
        pills(*[(s, "gray") for s in theme.get("scenarios", [])]) + "<br>" +
        "**Engagement Activity** " +
        pills(*[(activity_name(a), "gray")
                for a in theme.get("activity_ids", [])]),
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

    if st.button("Add Entire Theme to Decision",
                 key=f"themedec-{theme['theme_id']}", type="primary"):
        if theme["theme_id"] not in st.session_state.decision_staged_themes:
            st.session_state.decision_staged_themes.append(theme["theme_id"])
        st.toast(f'Theme “{theme["name"]}” staged for the next decision — '
                 'underlying evidence stays linked')

    st.markdown("---")
    st.markdown("**Evidence Within This Theme**")
    theme_ev = [e for e in st.session_state.evidence
                if e.get("theme_id") == theme["theme_id"]]
    quant_ev = [e for e in theme_ev
                if e["type"] in ("Quantitative Pattern", "Cross-Scenario Pattern")]
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
        fc1, fc2, fc3, fc4, fc5, fc6 = st.columns([1.3, 1.1, 1.1, 1, 1, 1.4])
        lf_act = fc1.selectbox("Engagement Activity", ["All"] + acts,
                               format_func=lambda a: a if a == "All"
                               else activity_name(a))
        lf_scen = fc2.selectbox(
            "Dataset / Scenario",
            ["All"] + sorted({s for t in themes for s in t.get("scenarios", [])}))
        lf_status = fc3.selectbox("Validation Status", ["All", "Human Validated"])
        lf_tag = fc4.selectbox("Tag", ["All"] + all_known_tags())
        lf_reac = fc5.selectbox("Reaction",
                                ["All", "approve", "disapprove", "none"])
        lf_q = fc6.text_input("Search Themes", placeholder="Search themes…")

        def theme_matches(t):
            if lf_act != "All" and lf_act not in t.get("activity_ids", []):
                return False
            if lf_scen != "All" and lf_scen not in t.get("scenarios", []):
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
                          *[(activity_name(a), "gray")
                            for a in theme.get("activity_ids", [])],
                          *[(s, "gray") for s in theme.get("scenarios", [])]),
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
                                st.session_state.themes.append({
                                    "theme_id": theme_id, "origin": "human",
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
# PAGE 04 — DECISION TRAILS
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
                'Inputs are <b>considered</b> — evidence does not automatically '
                'cause a decision.</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="ces-chain">{DECISION_DIAGRAM}</div>',
                unsafe_allow_html=True)

    theme_opts = {t["theme_id"]: f'{t["theme_id"]} · {t["name"]} '
                                 f'({len(t["record_ids"])} records)'
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
                        st.markdown(pills(("THEME", "validated"),
                                          (th["name"], "gray")) +
                                    f'<span class="ces-meta"> '
                                    f'{len(th["record_ids"])} records</span>',
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
    proj = st.session_state.project
    with st.sidebar:
        st.markdown("## Civic Evidence Studio")
        st.caption(f'{proj["metadata"]["project_name"]} · '
                   f'{proj["metadata"]["project_phase"]}')
        page = st.radio("Navigate", ["01 Data + Context", "02 Insights Playground",
                                     "03 Libraries", "04 Decision Trails"],
                        label_visibility="collapsed")
        st.divider()
        n_act = len(all_activities())
        n_ds = sum(len(a["datasets"]) for a in all_activities())
        processed = sum(1 for a in all_activities() if a["combined"] is not None)
        st.caption(
            f"Activities: {n_act} ({processed} processed)\n\n"
            f"Datasets: {n_ds}\n\n"
            f"Themes: {len(st.session_state.themes)}\n\n"
            f"Evidence items: {len(st.session_state.evidence)}\n\n"
            f"Constraints: {len(st.session_state.constraints)}\n\n"
            f"Decisions: {len(st.session_state.decisions)}")
        provider, _ = llm_provider()
        st.markdown(pill("AI: " + (provider if provider else "not configured"),
                         "ai" if provider else "review"), unsafe_allow_html=True)
        st.caption("PROJECT → ACTIVITY → DATASET → RECORD → THEME → EVIDENCE "
                   "→ DECISION. Provenance is preserved at every level.")

    if page == "01 Data + Context":
        page_data_context()
    elif page == "02 Insights Playground":
        page_insights()
    elif page == "03 Libraries":
        page_libraries()
    else:
        page_decisions()


main()
