import os
import sys

import pandas as pd
import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "duty_cycle"))

from bo_session import LOG_PATH  # noqa: E402
from ui_common import (  # noqa: E402
    PARAM_LABELS, get_gp_session, save_upload, process_image, structure_id_input, regret_chart,
)
import config  # noqa: E402

st.set_page_config(page_title="Duty cycle optimizer", page_icon=":material/biotech:", layout="wide")

session = get_gp_session()

st.title("Duty cycle optimizer")
st.caption(session.status_line())

st.subheader("Progress")
log_df = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0 else pd.DataFrame()
regret_chart(log_df, config.target_dc)

done, reason = session.is_finished()
if done:
    st.success(reason, icon=":material/check_circle:")
    st.stop()

params = session.next_params()

st.subheader("Recommended next experiment")
if session.pending_gp_fallback:
    st.warning(
        "The GP model failed to fit for this suggestion (a numerical issue -- the same rare "
        "kernel-singularity failure mode seen during evaluation), so a random point within bounds "
        "was used instead. This is logged (gp_fallback=True) so it can be filtered out later.",
        icon=":material/warning:",
    )
row = st.container(horizontal=True)
for label, value in zip(PARAM_LABELS, params):
    row.metric(label, f"{value:.2f}")
source_label = "Latin Hypercube warm-up" if session.pending_source == "lhs" else "GP + expected improvement"
st.caption(f"source: {source_label}")
if session.pending_prediction is not None:
    pred_mean, pred_std = session.pending_prediction
    st.caption(f"GP predicts {pred_mean * 100:.1f}% ± {pred_std * 100:.1f}% duty cycle here, "
               f"before measurement.")

st.subheader("Measure the fabricated structure")
angle = st.number_input(
    "Deskew angle (degrees)", value=0.0, step=0.1, format="%.2f",
    help="0.0 for the ZEHRA01-style batch (already horizontal). Adjust and re-check the "
         "boundary image below if the teeth look visibly tilted.",
)
uploaded = st.file_uploader("Microscope image", type=["tif", "tiff", "png", "jpg", "jpeg"])

if uploaded is not None:
    fingerprint = (uploaded.file_id, angle)
    if st.session_state.get("fingerprint") != fingerprint:
        saved_path = save_upload(uploaded, angle)
        with st.spinner("Detecting domain boundaries...", show_time=True):
            mean, std, diagnostics, boundary_png = process_image(saved_path, angle)
        st.session_state.fingerprint = fingerprint
        st.session_state.image_path = saved_path
        st.session_state.result = (mean, std, diagnostics, boundary_png)

    mean, std, diagnostics, boundary_png = st.session_state.result

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
        m.metric("Problem fraction", f"{diagnostics['problem_fraction']:.0%}" if diagnostics["problem_fraction"] is not None else "n/a")
        m.metric("Classification", diagnostics["classification"].replace("_", " "))

        st.markdown("**Boundary detection (visual check)**")
        st.caption("Magenta lines mark detected domain A/B boundaries; green ticks are validated "
                   "domain-A widths (red = periods in an underpoled region). Check these land on "
                   "real dark seams before submitting.")
        st.image(boundary_png, width="stretch")

        structure_id = structure_id_input(LOG_PATH)
        if not structure_id:
            st.caption(":material/info: Enter a structure ID above to enable submit.")

        if st.button("Confirm and submit", icon=":material/send:", type="primary", disabled=not structure_id):
            result = session.submit_result(structure_id, st.session_state.image_path, angle=angle)
            if result is None:
                st.error("Measurement changed on resubmit -- try uploading again.", icon=":material/error:")
            else:
                mean, std, output, diagnostics = result
                st.toast(f"Logged: {mean:.1f}% duty cycle, output={output:.4f}", icon=":material/check:")
                for key in ("fingerprint", "image_path", "result"):
                    st.session_state.pop(key, None)
                st.rerun()

st.subheader("Experiment log")
if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0:
    df = pd.read_csv(LOG_PATH)
    if len(df):
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button(
            "Download log CSV", data=df.to_csv(index=False), file_name="experiments.csv",
            mime="text/csv", icon=":material/download:",
        )
    else:
        st.caption("No experiments logged yet.")
else:
    st.caption("No experiments logged yet.")

best = session.best_result()
if best is not None:
    best_params, best_output = best
    st.caption(
        f"Best so far: output={best_output:.4f} at "
        + ", ".join(f"{label}={v:.2f}" for label, v in zip(PARAM_LABELS, best_params))
    )
