"""Shared Streamlit UI helpers -- image upload/storage, boundary-detection
preview, and the progress chart -- used by both streamlit_app.py (the main
single-arm tool) and pages/1_GP_vs_OFAT_Evaluation.py (the real-chip
comparison), so the two pages render identically instead of duplicating
this logic."""
import os
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

# bo_session MUST be imported before tip_lines_check: tip_lines_check prepends
# duty_cycle/algorithms to sys.path (it has its own unrelated config.py), and if that
# happens before gp_trial's own `config` module has been imported/cached anywhere in the
# process, a later `from config import ...` inside bo_session's own import chain
# (model.py/optimizer.py) silently resolves to the WRONG config.py and crashes with a
# confusing TypeError -- reproduced directly while testing this file in isolation.
# Importing bo_session first guarantees the real gp_trial config is cached before
# tip_lines_check's sys.path mutation can shadow it, regardless of what else imports
# this module or in what order.
from bo_session import BOSession
import tip_lines_check

PARAM_LABELS = ("Amplitude", "Rise", "Dwell", "Fall")

# every uploaded image is saved here permanently, immediately on upload -- not just on confirm --
# so a rejected/unmeasurable image or one abandoned mid-angle-adjustment is never lost, and so the
# logged image_path always points somewhere durable instead of an OS temp file that can be cleaned
# up independently of the CSV row that references it
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


@st.cache_resource
def get_gp_session():
    """The ONE shared BOSession for the whole server process -- both
    streamlit_app.py and the evaluation page must import this (not define
    their own @st.cache_resource BOSession getter), otherwise Streamlit
    caches per-function and the two pages end up with two independent,
    silently-diverging session objects reading/writing the same CSV."""
    return BOSession()


def save_upload(uploaded, angle):
    suffix = os.path.splitext(uploaded.name)[1] or ".tif"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(uploaded.name)[0]
    path = os.path.join(IMAGES_DIR, f"{stamp}_{stem}_angle{angle:g}_{uuid.uuid4().hex[:6]}{suffix}")
    with open(path, "wb") as f:
        f.write(uploaded.getvalue())
    return path


def process_image(path, angle):
    """(mean, std, diagnostics, boundary_png_path) or (None, None, None, None)
    if no numeric duty cycle could be computed. Exceptions from a malformed
    image are caught and surfaced as a diagnostics-shaped error instead of
    crashing the app."""
    try:
        mean, std, diagnostics = tip_lines_check.estimate_duty_cycle(path, angle=angle)
        boundary_png = tip_lines_check.draw_domain_boundary_lines(path, angle=angle, upscale=2)
        return mean, std, diagnostics, boundary_png
    except Exception as e:
        st.error(f"Could not process this image: {e}", icon=":material/error:")
        return None, None, None, None


def structure_id_input(log_path, key=None):
    """Text input for the physical structure ID, with a duplicate warning --
    the code has no way to know the chip's layout, so this is typed in, not
    auto-assigned. Returns the entered ID (empty string if none yet)."""
    from bo_session import used_structure_ids
    structure_id = st.text_input(
        "Physical structure ID",
        help="The real structure/location label on your chip -- e.g. its row/column on the test "
             "pattern. Not auto-assigned; the code doesn't know your chip's physical layout.",
        key=key,
    )
    if structure_id and structure_id in used_structure_ids(log_path):
        st.warning(
            f"Structure '{structure_id}' is already logged. If you're correcting a photo of the "
            f"same structure, that's fine -- otherwise double-check the ID.",
            icon=":material/warning:",
        )
    return structure_id


def regret_chart(log_df, target_dc, split_by_source=False):
    """Builds and renders the best-so-far |duty cycle - target| chart from
    the experiment log. split_by_source=True draws a separate "best so far"
    series per distinct `source` value (e.g. 'gp_ei' vs 'ofat'), each
    indexed by THAT source's own count of real structures used -- the
    matched-budget comparison, not overall row order."""
    if not len(log_df) or "duty_cycle_mean" not in log_df.columns:
        st.caption("No real experiments logged yet -- submit your first measurement to start tracking progress.")
        return
    real_rows = log_df.dropna(subset=["duty_cycle_mean"])
    if not len(real_rows):
        st.caption("No real experiments logged yet -- submit your first measurement to start tracking progress.")
        return

    if not split_by_source:
        regret = (real_rows["duty_cycle_mean"] / 100 - target_dc).abs()
        chart_df = pd.DataFrame({
            "regret (this experiment)": regret.values,
            "best so far": regret.cummin().values,
        }, index=range(1, len(regret) + 1))
        if "gp_pred_mean" in real_rows.columns and real_rows["gp_pred_mean"].notna().any():
            chart_df["GP predicted regret"] = (real_rows["gp_pred_mean"] - target_dc).abs().values
        st.line_chart(chart_df)
        st.caption(f"{len(real_rows)} real experiment(s) logged. Regret = |duty cycle - "
                   f"{target_dc:.0%}| (lower is better).")
    else:
        series = {}
        for source, group in real_rows.groupby("source"):
            regret = (group["duty_cycle_mean"] / 100 - target_dc).abs()
            series[f"{source}: best so far"] = regret.cummin().values
        chart_df = pd.DataFrame({
            name: pd.Series(values, index=range(1, len(values) + 1))
            for name, values in series.items()
        })
        chart_df.index.name = "structures used (this arm)"
        st.line_chart(chart_df)
        st.caption(f"Regret = |duty cycle - {target_dc:.0%}| (lower is better), indexed by each "
                   f"arm's own structure count (matched-budget comparison, not overall row order).")
