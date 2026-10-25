"""GP+EI vs. OFAT live comparison on the real chip -- a second, self-
contained Streamlit page. Both arms draw from the same chip, alternating
structure-by-structure, sharing the one experiments.csv log (tagged
source='gp_ei' or source='ofat'). Doesn't touch the main single-arm flow in
streamlit_app.py -- see ui_common.get_gp_session for why both pages share
the exact same cached BOSession instance rather than each caching their own."""
import os
import sys

import pandas as pd
import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "duty_cycle"))

from bo_session import LOG_PATH  # noqa: E402
from ofat_session import OFATSession  # noqa: E402
from ui_common import (  # noqa: E402
    PARAM_LABELS, get_gp_session, save_upload, process_image, structure_id_input, regret_chart,
)
import config  # noqa: E402

st.set_page_config(page_title="GP vs OFAT evaluation", page_icon=":material/science:", layout="wide")

N_CANDIDATES = 4
N_CYCLES = 3
# 1 shared starting-point evaluation (OFAT's history[0], same params as GP's first suggestion) +
# N_CYCLES full sweeps of all 4 parameters at N_CANDIDATES each -- see ofat.ofat_next_candidate
BUDGET_PER_ARM = 1 + N_CYCLES * 4 * N_CANDIDATES
TOTAL_STRUCTURES = 145

st.title("GP + EI vs. OFAT -- real chip comparison")
st.caption(
    f"Matched budget: {BUDGET_PER_ARM} real structures per arm ({BUDGET_PER_ARM * 2} of "
    f"{TOTAL_STRUCTURES} committed, rest held as buffer). Both arms always run their full budget, "
    f"even after a good result, so the comparison isn't confounded by different stopping points."
)

gp_session = get_gp_session()
gp_start = gp_session.next_params()  # GP's own first suggestion -- also OFAT's shared starting point
ofat_session = OFATSession(gp_session.bounds, gp_start, n_candidates=N_CANDIDATES)

log_df = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0 else pd.DataFrame()
gp_count = int((log_df["source"] == "gp_ei").sum()) if len(log_df) and "source" in log_df.columns else 0
ofat_count = int((log_df["source"] == "ofat").sum()) if len(log_df) and "source" in log_df.columns else 0

st.subheader("Progress")
regret_chart(log_df, config.target_dc, split_by_source=True)
c1, c2 = st.columns(2)
c1.metric("GP + EI arm", f"{gp_count} / {BUDGET_PER_ARM}")
c2.metric("OFAT arm", f"{ofat_count} / {BUDGET_PER_ARM}")

gp_done = gp_count >= BUDGET_PER_ARM
ofat_done = ofat_count >= BUDGET_PER_ARM
if gp_done and ofat_done:
    st.success("Both arms have used their full budget -- comparison complete.", icon=":material/check_circle:")
    st.stop()

# Alternating schedule: whichever arm has fewer structures so far goes next (GP breaks ties), so
# the two arms never drift more than one structure apart over the session.
if not gp_done and gp_count <= ofat_count:
    suggested_arm = "GP + EI"
elif not ofat_done:
    suggested_arm = "OFAT"
else:
    suggested_arm = "GP + EI"

arm_options = [a for a, done in (("GP + EI", gp_done), ("OFAT", ofat_done)) if not done]
arm = st.radio("Which arm are you submitting for?", arm_options,
                index=arm_options.index(suggested_arm), horizontal=True)
if arm != suggested_arm:
    st.caption(f"Note: the alternating schedule suggests **{suggested_arm}** next.")

session = gp_session if arm == "GP + EI" else ofat_session
params = session.next_params()

st.subheader("Recommended next experiment")
if arm == "GP + EI" and session.pending_gp_fallback:
    st.warning(
        "The GP model failed to fit for this suggestion (a numerical issue -- the same rare "
        "kernel-singularity failure mode seen during evaluation), so a random point within bounds "
        "was used instead. This is logged (gp_fallback=True).",
        icon=":material/warning:",
    )
row = st.container(horizontal=True)
for label, value in zip(PARAM_LABELS, params):
    row.metric(label, f"{value:.2f}")
st.caption(f"arm: {arm}")
if arm == "GP + EI" and session.pending_prediction is not None:
    pred_mean, pred_std = session.pending_prediction
    st.caption(f"GP predicts {pred_mean * 100:.1f}% ± {pred_std * 100:.1f}% duty cycle here, "
               f"before measurement.")
elif arm == "OFAT":
    info = session.pending_info
    if info["is_baseline"]:
        st.caption("OFAT's shared starting point -- same parameters as GP's first suggestion.")
    else:
        st.caption(f"Sweeping {PARAM_LABELS[info['param_index']]} "
                   f"(candidate {info['candidate_index'] + 1}/{N_CANDIDATES}, step {info['step'] + 1})"
                   + (" -- new cycle" if info["is_new_cycle"] else ""))

st.subheader("Measure the fabricated structure")
structure_id = structure_id_input(LOG_PATH, key="eval_structure_id")
angle = st.number_input(
    "Deskew angle (degrees)", value=0.0, step=0.1, format="%.2f",
    help="0.0 if already horizontal. Adjust and re-check the boundary image below if the teeth "
         "look visibly tilted.",
)
uploaded = st.file_uploader("Microscope image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="eval_uploader")

if uploaded is not None:
    fingerprint = (uploaded.file_id, angle, arm)
    if st.session_state.get("eval_fingerprint") != fingerprint:
        saved_path = save_upload(uploaded, angle)
        with st.spinner("Detecting domain boundaries...", show_time=True):
            mean, std, diagnostics, boundary_png = process_image(saved_path, angle)
        st.session_state.eval_fingerprint = fingerprint
        st.session_state.eval_image_path = saved_path
        st.session_state.eval_result = (mean, std, diagnostics, boundary_png)

    mean, std, diagnostics, boundary_png = st.session_state.eval_result

    if mean is None:
        if diagnostics is not None and diagnostics.get("n_domain_a_found", 1) <= 1:
            st.warning(
                "No domain-A boundaries could be paired anywhere in this image -- consistent with "
                "severe overpoling (domains merged into each other) rather than a processing failure.",
                icon=":material/warning:",
            )
        else:
            st.warning("No numeric duty cycle could be computed from this image. Try a different one.",
                       icon=":material/warning:")
    else:
        m = st.container(horizontal=True)
        m.metric("Mean duty cycle", f"{mean:.1f}%")
        m.metric("Std dev", f"{std:.1f}%")
        m.metric("Problem fraction",
                 f"{diagnostics['problem_fraction']:.0%}" if diagnostics["problem_fraction"] is not None else "n/a")
        m.metric("Classification", diagnostics["classification"].replace("_", " "))

        st.markdown("**Boundary detection (visual check)**")
        st.caption("Magenta lines mark detected domain A/B boundaries; green ticks are validated "
                   "domain-A widths (red = periods in an underpoled region). Check these land on "
                   "real dark seams before submitting.")
        st.image(boundary_png, width="stretch")
        if not structure_id:
            st.caption(":material/info: Enter a structure ID above to enable submit.")

        if st.button("Confirm and submit", icon=":material/send:", type="primary", disabled=not structure_id):
            result = session.submit_result(structure_id, st.session_state.eval_image_path, angle=angle)
            if result is None:
                st.error("Measurement changed on resubmit -- try uploading again.", icon=":material/error:")
            else:
                mean, std, output, diagnostics = result
                st.toast(f"Logged ({arm}): {mean:.1f}% duty cycle", icon=":material/check:")
                for key in ("eval_fingerprint", "eval_image_path", "eval_result"):
                    st.session_state.pop(key, None)
                st.rerun()

st.subheader("Experiment log")
if len(log_df):
    st.dataframe(log_df, width="stretch", hide_index=True)
    st.download_button(
        "Download log CSV", data=log_df.to_csv(index=False), file_name="experiments.csv",
        mime="text/csv", icon=":material/download:",
    )
else:
    st.caption("No experiments logged yet.")
