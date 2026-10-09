import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
SYM = {"+": "+", "-": "−", "*": "×", "/": "÷", "=": "", "?": "?"}

st.set_page_config(page_title="KenKen Solver", page_icon="🧩", layout="centered")
st.title(" KenKen Solver")
st.caption("Take a photo of the board, check what the model read, then solve it.")


def draw_board(puzzle, solution=None):
    n = puzzle["n"]
    owner = {}
    for i, cg in enumerate(puzzle["cages"]):
        for r, c in cg["cells"]:
            owner[(r, c)] = i
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, n); ax.set_ylim(n, 0); ax.set_aspect("equal"); ax.axis("off")
    for k in range(n + 1):                      # thin grid
        ax.plot([0, n], [k, k], color="#bbb", lw=0.8)
        ax.plot([k, k], [0, n], color="#bbb", lw=0.8)
    for r in range(n):                          # thick cage borders
        for c in range(n):
            o = owner.get((r, c))
            if c == n - 1 or owner.get((r, c + 1)) != o:
                ax.plot([c + 1, c + 1], [r, r + 1], color="k", lw=3)
            if c == 0 or owner.get((r, c - 1)) != o:
                ax.plot([c, c], [r, r + 1], color="k", lw=3)
            if r == n - 1 or owner.get((r + 1, c)) != o:
                ax.plot([c, c + 1], [r + 1, r + 1], color="k", lw=3)
            if r == 0 or owner.get((r - 1, c)) != o:
                ax.plot([c, c + 1], [r, r], color="k", lw=3)
    for cg in puzzle["cages"]:
        r, c = min(map(tuple, cg["cells"]))
        bad = cg["op"] == "?" or cg["target"] < 0
        label = "?" if bad else f"{cg['target']}{SYM.get(cg['op'], cg['op'])}"
        ax.text(c + 0.06, r + 0.08, label, va="top", ha="left", fontsize=11,
                color="red" if bad else "black", weight="bold")
    if solution:
        for r in range(n):
            for c in range(n):
                ax.text(c + 0.5, r + 0.58, str(solution[r][c]), ha="center",
                        va="center", fontsize=22, color="#1f5fbf")
    return fig


def call_api(path, **kw):
    try:
        r = requests.post(f"{API_URL}{path}", timeout=120, **kw)
    except requests.RequestException as e:
        st.error(f"Cannot reach the API: {e}")
        return None
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:                       # body is not JSON
            detail = f"HTTP {r.status_code}: {r.text[:300]}"
        st.error(detail)
        return None
    return r.json()

# ---- input ---------------------------------------------------------------------
tab_cam, tab_up = st.tabs([" Camera", " Upload"])
with tab_cam:
    shot = st.camera_input("Fit the whole board inside the frame")
with tab_up:
    up = st.file_uploader("Board image", type=["png", "jpg", "jpeg"])
img = shot or up

if img is not None and st.button(" Read board", type="primary"):
    with st.spinner("Processing image..."):
        res = call_api("/parse", files={"file": (img.name if hasattr(img, "name") else "photo.jpg",
                                                 img.getvalue(), "image/jpeg")})
    if res:
        st.session_state.puzzle = {"n": res["n"], "cages": res["cages"]}
        st.session_state.solution = None
        if res.get("unreadable"):
            st.warning(f"{len(res['unreadable'])} clue(s) could not be read (shown as ? in red). Fix them below.")

# ---- review / edit / solve ----------------------------------------------------------
if "puzzle" in st.session_state:
    puzzle = st.session_state.puzzle
    st.subheader(f"Detected board: {puzzle['n']}×{puzzle['n']}, {len(puzzle['cages'])} cages")
    st.pyplot(draw_board(puzzle, st.session_state.get("solution")))

    with st.expander(" Correct the JSON if the model misread something"):
        txt = st.text_area("puzzle JSON", json.dumps(puzzle), height=220)
        if st.button("Apply changes"):
            try:
                st.session_state.puzzle = json.loads(txt)
                st.session_state.solution = None
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    if st.button("Solve"):
        with st.spinner("Solving..."):
            out = call_api("/solve", json=puzzle)
        if out:
            st.session_state.solution = out["solution"]
            st.rerun()

    st.download_button(" Download JSON", json.dumps(puzzle, indent=1), "puzzle.json")