import json
import os
from typing import List, Dict, Any

import streamlit as st

st.set_page_config(page_title="Data Labeler", layout="wide")

# --- State init (must run before any access) ---
def _init_state():
    ss = st.session_state
    if "_dataset" not in ss:
        ss._dataset = None
    if "_adding_abstraction" not in ss:
        ss._adding_abstraction = False
    if "_abstraction_step" not in ss:
        ss._abstraction_step = 0
    if "_abstraction" not in ss:
        ss._abstraction = {
            "name": None,
            "label": None,
            "trigger": None,
            "text_span": None,
            "word_start": None,
            "word_end": None,
            "procedure": [],
        }

_init_state()

# --- Header ---
st.markdown("# Data Labeler")
header_left, header_right = st.columns([1, 3])
with header_left:
    add_clicked = st.button("Add New Abstraction", use_container_width=True)
with header_right:
    st.write("")  # spacer

# Handle header button
if add_clicked:
    st.session_state._adding_abstraction = True
    st.session_state._abstraction_step = 0
    st.rerun()

# --- File input ---
left, right = st.columns([2, 1])
with left:
    uploaded = st.file_uploader("Upload a JSON file (list[dict])", type=["json"], accept_multiple_files=False)
with right:
    file_path = st.text_input("Or enter a JSON file path", value=st.session_state.get("_last_path", ""))

# Keep data stable across reruns
if "_dataset" not in st.session_state:
    st.session_state._dataset = None

# Prefer explicit path if provided; else use uploaded file
raw_data = None
if file_path.strip():
    if not os.path.exists(file_path):
        st.error(f"Path not found: {file_path}")
        st.stop()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        st.session_state["_last_path"] = file_path
    except Exception as e:
        st.exception(e)
        st.stop()
elif uploaded is not None:
    try:
        raw_data = json.load(uploaded)
    except Exception as e:
        st.exception(e)
        st.stop()

if raw_data is not None:
    if not isinstance(raw_data, list) or not all(isinstance(x, dict) for x in raw_data):
        st.error("JSON must be a list of dictionaries.")
        st.stop()
    
    raw_data = [r for r in raw_data if r['solver_correct']]
    st.session_state._dataset = raw_data


data: List[Dict[str, Any]] | None = st.session_state._dataset

if data is None:
    st.info("Upload a JSON file to begin.")
    st.stop()

# --- Helpers ---
CANDIDATE_KEYS = ("id", "name", "title", "uuid", "key")

def make_label(i: int, row: Dict[str, Any]) -> str:
    for k in CANDIDATE_KEYS:
        if k in row and isinstance(row[k], (str, int, float)):
            return f"{i}: {row[k]}"
    # Fallback: show first few keys to help identify
    preview = ", ".join(list(row.keys())[:3]) or "entry"
    return f"{i}: {preview}"

# --- Selection UI ---
options = [make_label(i, row) for i, row in enumerate(data)]
selected_label = st.selectbox("Select entry", options, index=0)
selected_idx = options.index(selected_label)
selected = data[selected_idx]

# --- Display target field ---
st.subheader("solver_full_output")
solver_out = selected.get("solver_full_output", "<missing>")
st.text_area(
    label="solver_full_output",
    value=str(solver_out),
    height=320,
    label_visibility="collapsed",
    disabled=True,
)

# --- Optional: context panel ---
with st.expander("Show full entry (JSON)"):
    st.json(selected, expanded=False)

# The "Add New Abstraction" button currently does nothing by design; we'll wire it up next.


# --- Abstraction Wizard -----------------------------------------------------
if st.session_state.get("_adding_abstraction", False):
    st.divider()
    st.markdown("### Abstraction Builder")

    # Pull current selection and solver text
    options = [make_label(i, row) for i, row in enumerate(data)]
    selected_label = st.session_state.get("entry_select", options[0]) if options else None
    selected_idx = options.index(selected_label) if selected_label in options else 0
    selected = data[selected_idx]
    solver_out = selected.get("solver_full_output", "<missing>")

    left_col, right_col = st.columns([1, 1])

    with right_col:
        st.subheader("solver_full_output")
        st.text_area(
            label="solver_full_output",
            value=str(solver_out),
            height=440,
            label_visibility="collapsed",
            disabled=True,
            key="solver_text_locked2",
        )

    with left_col:
        st.subheader("Define Abstraction")
        step = st.session_state._abstraction_step
        abstraction = st.session_state._abstraction

        def nav_buttons(show_back: bool = True):
            cols = st.columns([1,1,3])
            with cols[0]:
                if show_back and st.button("Back", use_container_width=True, key=f"back_{step}"):
                    st.session_state._abstraction_step = max(0, step - 1)
                    st.rerun()
            with cols[1]:
                if st.button("Next", use_container_width=True, key=f"next_{step}"):
                    st.session_state._abstraction_step = step + 1
                    st.rerun()

        # Step 0: Name
        if step == 0:
            name = st.text_input("Abstraction name", value=abstraction.get("name") or "")
            if name:
                abstraction["name"] = name
            nav_buttons(show_back=False)

        # Step 1: Label (tag)
        elif step == 1:
            DEFAULT_TAGS = [
                "algebraic",
                "combinatorial",
                "number_theoretic",
                "geometric",
                "inequality",
                "search_strategy",
                "sanity_check",
                "meta_control",
            ]
            if "_tags" not in st.session_state:
                st.session_state._tags = DEFAULT_TAGS.copy()
            tag_options = st.session_state._tags + ["➕ Add new tag…"]
            current = abstraction.get("label") if abstraction.get("label") in tag_options else None
            idx = tag_options.index(current) if current else 0
            choice = st.selectbox("Label", tag_options, index=idx)
            if choice == "➕ Add new tag…":
                new_tag = st.text_input("New tag name")
                if new_tag:
                    if new_tag not in st.session_state._tags:
                        st.session_state._tags.append(new_tag)
                    abstraction["label"] = new_tag
                    st.success(f"Added tag '{new_tag}' and selected it.")
            else:
                abstraction["label"] = choice
            nav_buttons(show_back=True)

        # Step 2: Trigger text
        elif step == 2:
            trig = st.text_area(
                "Trigger — textual cue/structural conditions indicating applicability",
                value=abstraction.get("trigger") or "",
                height=120,
            )
            if trig:
                abstraction["trigger"] = trig
            nav_buttons(show_back=True)

        # Step 3: Text span selection helper
        elif step == 3:
            st.markdown("**Select the text span** — for now, paste the exact span or set word indices; direct highlighting will be added.")
            full_text = str(solver_out)

            def compute_word_positions(text: str):
                positions = []  # list of (start_char, end_char)
                i = 0
                n = len(text)
                while i < n:
                    if text[i].isspace():
                        i += 1
                        continue
                    j = i
                    while j < n and not text[j].isspace():
                        j += 1
                    positions.append((i, j))
                    i = j
                return positions

            positions_cache = compute_word_positions(full_text)

            def indices_from_span(span: str):
                if not span:
                    return None
                start_char = full_text.find(span)
                if start_char == -1:
                    return None
                end_char = start_char + len(span)
                w_start = None
                w_end = None
                for wi, (a, b) in enumerate(positions_cache):
                    if b <= start_char:
                        continue
                    if a >= end_char and w_start is not None:
                        break
                    if w_start is None and b > start_char:
                        w_start = wi
                    if a < end_char:
                        w_end = wi
                if w_start is None or w_end is None:
                    return None
                return w_start, w_end

            c1, c2 = st.columns(2)
            with c1:
                span_text = st.text_area("Selected text (paste exact)", value=abstraction.get("text_span") or "", height=140)
                if st.button("Capture span from text", key="cap_span2"):
                    res = indices_from_span(span_text)
                    if res is None:
                        st.error("Span not found in solver output.")
                    else:
                        w0, w1 = res
                        abstraction["text_span"] = span_text
                        abstraction["word_start"] = int(w0)
                        abstraction["word_end"] = int(w1)
                        st.success(f"Captured word indices: {w0}–{w1}")
            with c2:
                w_start_val = int(abstraction.get("word_start") or 0)
                w_end_val = int(abstraction.get("word_end") or 0)
                w_start = st.number_input("Word start index", min_value=0, value=w_start_val, step=1)
                w_end = st.number_input("Word end index", min_value=0, value=w_end_val, step=1)
                if st.button("Set indices manually", key="set_idx2"):
                    abstraction["word_start"] = int(w_start)
                    abstraction["word_end"] = int(w_end)
                    if 0 <= w_start < len(positions_cache) and 0 <= w_end < len(positions_cache) and w_end >= w_start:
                        a, _ = positions_cache[w_start]
                        _, b = positions_cache[w_end]
                        abstraction["text_span"] = full_text[a:b]
                        st.success("Derived span from indices.")

            nav_buttons(show_back=True)

        # Step 4: Procedure (multi-step recipe)
        elif step == 4:
            st.markdown("**Procedure** — add ordered steps for implementing the abstraction.")
            procedure: list[str] = abstraction.get("procedure") or []

            # Ensure we have at least one step slot
            if len(procedure) == 0:
                procedure = [""]

            # Render editable areas for each step
            new_procedure = []
            for i, text in enumerate(procedure):
                val = st.text_area(f"Step {i+1}", value=text, height=90, key=f"proc_{i}")
                new_procedure.append(val)

            cadd, cdel, _ = st.columns([1,1,2])
            with cadd:
                if st.button("Add step", key="add_proc_step"):
                    new_procedure.append("")
                    abstraction["procedure"] = new_procedure
                    st.rerun()
            with cdel:
                if st.button("Remove last", key="del_proc_step") and len(new_procedure) > 1:
                    new_procedure = new_procedure[:-1]
                    abstraction["procedure"] = new_procedure
                    st.rerun()

            abstraction["procedure"] = new_procedure
            nav_buttons(show_back=True)

        # Persist abstraction state
        st.session_state._abstraction = abstraction
        st.divider()
        if st.button("Finish Later / Close", key="close_wizard2"):
            st.session_state._adding_abstraction = False
            st.rerun()
