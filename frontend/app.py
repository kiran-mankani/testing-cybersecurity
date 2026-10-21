"""Streamlit frontend for the intrusion-detection API (backend/main.py).

Run from the project root, with the API already running:
    streamlit run frontend/app.py
"""
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
BATCH_CHUNK = 5000

# NSL-KDD feature groups (basic, content, time-based traffic, host-based traffic)
GROUPS = {
    "Basic connection": ["duration", "protocol_type", "service", "flag", "src_bytes",
                         "dst_bytes", "land", "wrong_fragment", "urgent"],
    "Content": ["hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
                "su_attempted", "num_root", "num_file_creations", "num_shells",
                "num_access_files", "is_host_login", "is_guest_login"],
    "Traffic (same host / service)": ["count", "srv_count", "serror_rate", "srv_serror_rate",
                                      "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                                      "diff_srv_rate", "srv_diff_host_rate"],
    "Host-based traffic": ["dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                           "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                           "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                           "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                           "dst_host_srv_rerror_rate"],
}
BINARY = {"land", "logged_in", "is_host_login", "is_guest_login"}
VERDICT_STYLE = {"Normal": ("ok", "Normal traffic"),
                 "Attack": ("bad", "Attack detected"),
                 "Anomaly": ("warn", "Anomalous traffic")}

st.set_page_config(page_title="Intrusion Detection", page_icon="🛡️", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>",
            unsafe_allow_html=True)


# ---------------------------------------------------------------- API helpers
def api(method, path, **kwargs):
    try:
        r = requests.request(method, API + path, timeout=kwargs.pop("timeout", 30), **kwargs)
    except requests.ConnectionError:
        return None, f"Cannot reach the API at {API}. Start it with: uvicorn backend.main:app"
    if r.ok:
        return r.json(), None
    try:
        detail = r.json().get("detail")
    except ValueError:
        detail = r.text
    return None, f"API error {r.status_code}: {detail}"


@st.cache_data(ttl=60)
def model_info():
    return api("GET", "/model/info")


def is_rate(name):
    return name.endswith("_rate")


# ---------------------------------------------------------------- sidebar
health, health_err = api("GET", "/health", timeout=5)
with st.sidebar:
    st.markdown("### 🛡️ Intrusion Detection")
    st.caption("NSL-KDD · XGBoost + Isolation Forest")
    page = st.radio("Page", ["Predict", "Batch CSV", "History", "Model performance", "About"],
                    label_visibility="collapsed")
    st.divider()
    if health_err:
        st.error(health_err)
    else:
        st.markdown(("🟢" if health["model_loaded"] else "🔴") + " Model: "
                    + (health["model"] or "not loaded"))
        st.markdown(("🟢" if health["database_connected"] else "🟠") + " MongoDB: "
                    + ("connected" if health["database_connected"] else "offline (history disabled)"))
        if health["model_error"]:
            st.caption(health["model_error"])

info, info_err = model_info()


def need_model():
    if info_err:
        st.error(info_err)
        st.stop()


# ---------------------------------------------------------------- result card
def show_result(pred, elapsed_ms=None, stored=None):
    cls, title = VERDICT_STYLE[pred["verdict"]]
    st.markdown(
        f"<div class='verdict {cls}'><div class='v-title'>{title}</div>"
        f"<div class='v-class'>{pred['predicted_class']}</div>"
        f"<div class='v-sub'>Classifier confidence {pred['confidence']:.1%} · "
        f"Severity <b>{pred['severity']}</b> · Response <b>{pred['response']}</b></div></div>",
        unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Classifier says", "Attack" if pred["classifier_says_attack"] else "Normal")
    c2.metric("Anomaly score", f"{pred['anomaly_score']:.3f}",
              f"threshold {pred['anomaly_threshold']:.3f}", delta_color="off")
    c3.metric("Isolation Forest", "Anomalous" if pred["anomaly_flag"] else "Within normal")
    top = pd.DataFrame(pred["top_classes"])
    fig = px.bar(top[::-1], x="probability", y="class", orientation="h", text_auto=".3f",
                 range_x=[0, 1], height=230)
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), xaxis_title="Probability", yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    for w in pred["warnings"]:
        st.warning(w)
    notes = []
    if elapsed_ms is not None:
        notes.append(f"{elapsed_ms:.1f} ms")
    if stored is not None:
        notes.append("saved to MongoDB" if stored else "not saved (MongoDB offline)")
    if notes:
        st.caption(" · ".join(notes))


# ---------------------------------------------------------------- pages
def page_predict():
    need_model()
    st.title("Classify a network connection")
    st.caption("Enter the 40 NSL-KDD features, or load a real record from data/raw/kdd_test.csv.")

    ex, ex_err = api("GET", "/examples", params={"n": 1})
    labels = ex["labels_available"] if ex else []

    def load_example():
        params = {"n": 1}
        if st.session_state.example_label != "any":
            params["label"] = st.session_state.example_label
        data, err = api("GET", "/examples", params=params)
        if err:
            st.session_state.example_error = err
            return
        rec = data["records"][0]
        st.session_state.true_label = rec.pop("labels")
        for k, v in rec.items():
            st.session_state[k] = v if k in info["categorical_features"] else float(v)

    c1, c2, c3 = st.columns([2, 1, 3])
    c1.selectbox("Example label", ["any"] + labels, key="example_label")
    c2.button("Load example", on_click=load_example, disabled=ex_err is not None,
              width="stretch")
    if st.session_state.get("true_label"):
        c3.info(f"Loaded test record — true label: **{st.session_state.true_label}**")
    if st.session_state.get("example_error"):
        st.error(st.session_state.pop("example_error"))

    with st.form("features"):
        tabs = st.tabs(list(GROUPS))
        for tab, names in zip(tabs, GROUPS.values()):
            cols = tab.columns(3)
            for i, name in enumerate(names):
                col = cols[i % 3]
                if name in info["categorical_features"]:
                    options = info["categories"][name]
                    if st.session_state.get(name) not in options:
                        st.session_state[name] = options[0]
                    col.selectbox(name, options, key=name)
                elif name in BINARY:
                    st.session_state.setdefault(name, 0.0)
                    col.selectbox(name, [0.0, 1.0], key=name, format_func=lambda v: str(int(v)))
                else:
                    st.session_state.setdefault(name, 0.0)
                    col.number_input(name, min_value=0.0, max_value=1.0 if is_rate(name) else None,
                                     step=0.01 if is_rate(name) else 1.0, key=name)
        store = st.checkbox("Save prediction to MongoDB", value=True)
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if submitted:
        payload = {name: st.session_state[name] for name in info["feature_names"]}
        res, err = api("POST", "/predict", json=payload, params={"store": store})
        if err:
            st.error(err)
        else:
            st.subheader("Result")
            show_result(res["prediction"], res["elapsed_ms"], res["stored"] if store else None)
            truth = st.session_state.get("true_label")
            if truth:
                st.caption(f"True label of the loaded record: {truth}"
                           + ("" if truth in info["classes"] else
                              " — this attack type is not among the 27 training classes"))


def page_batch():
    need_model()
    st.title("Score a CSV file")
    st.caption("The CSV needs the 40 feature columns (extra columns are ignored). "
               "If it has a `labels` column, the page also compares predictions with it.")
    file = st.file_uploader("CSV file", type="csv")
    store = st.checkbox("Save predictions to MongoDB", value=False)
    if not file:
        return
    df = pd.read_csv(file)
    missing = [c for c in info["feature_names"] if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return
    st.write(f"{len(df):,} rows loaded.")
    if not st.button("Run predictions", type="primary"):
        return

    preds, bar = [], st.progress(0.0)
    for start in range(0, len(df), BATCH_CHUNK):
        chunk = df.iloc[start:start + BATCH_CHUNK][info["feature_names"]]
        res, err = api("POST", "/predict/batch", timeout=300, json={
            "records": chunk.to_dict(orient="records"), "store": store, "source": "csv"})
        if err:
            st.error(err)
            return
        preds += res["predictions"]
        bar.progress(min(1.0, (start + BATCH_CHUNK) / len(df)))

    out = df.copy()
    for key in ("predicted_class", "confidence", "verdict", "severity", "response",
                "anomaly_score", "alert"):
        out[key] = [p[key] for p in preds]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(out):,}")
    c2.metric("Alerts", f"{int(out['alert'].sum()):,}")
    c3.metric("Anomaly-only alerts", f"{int((out['verdict'] == 'Anomaly').sum()):,}")
    if "labels" in out.columns:
        known = out["labels"].isin(info["classes"])
        is_attack = out["labels"] != "normal"
        st.markdown(
            f"**Against the `labels` column:** exact-class accuracy on known classes "
            f"{(out.loc[known, 'labels'] == out.loc[known, 'predicted_class']).mean():.2%} "
            f"({known.sum():,} rows) · attacks alerted {out.loc[is_attack, 'alert'].mean():.2%} "
            f"· normal rows alerted (false alarms) {out.loc[~is_attack, 'alert'].mean():.2%}")
    counts = out["predicted_class"].value_counts().rename_axis("class").reset_index(name="rows")
    st.plotly_chart(px.bar(counts, x="class", y="rows", height=300), width="stretch")
    st.dataframe(out, width="stretch", height=350)
    st.download_button("Download predictions CSV", out.to_csv(index=False), "predictions.csv",
                       "text/csv")


def page_history():
    st.title("Prediction history")
    stats, err = api("GET", "/predictions/stats")
    if err:
        st.warning(err)
        return
    c1, c2 = st.columns(2)
    c1.metric("Stored predictions", f"{stats['total']:,}")
    c2.metric("Alerts", f"{stats['alerts']:,}")
    if stats["by_class"]:
        st.plotly_chart(px.bar(pd.DataFrame(stats["by_class"]), x="predicted_class", y="count",
                               height=300), width="stretch")
    limit = st.slider("Show latest", 10, 500, 50, step=10)
    rows, err = api("GET", "/predictions", params={"limit": limit})
    if err:
        st.warning(err)
    elif rows:
        table = pd.DataFrame(rows)[["timestamp", "source", "predicted_class", "confidence",
                                    "verdict", "severity", "response", "anomaly_flag"]]
        st.dataframe(table, width="stretch", hide_index=True)
    else:
        st.info("No predictions stored yet.")


def page_performance():
    st.title("Model performance")
    perf, err = api("GET", "/model/performance")
    if err:
        st.error(err)
        return
    st.caption("Read from the result files written by src/evaluation/ — nothing is recomputed here. "
               "reports/final_report.md quotes an earlier run of the same script; "
               "see the project report, section 6.4.")
    if "cv_summary" in perf:
        st.subheader("5-fold CV on deduplicated NSL-KDD")
        df = pd.DataFrame(perf["cv_summary"]["rows"]).set_index("model")
        st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns if df[c].dtype == float}),
                     width="stretch")
        st.caption(perf["cv_summary"]["source"])
        long = df[["missed_attacks", "false_alarms"]].reset_index().melt("model")
        st.plotly_chart(px.bar(long, x="model", y="value", color="variable", barmode="group",
                               height=320), width="stretch")
    if "hybrid_official_split" in perf:
        st.subheader("Official NSL-KDD split — attack types unseen in training")
        df = pd.DataFrame(perf["hybrid_official_split"]["rows"])
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption(perf["hybrid_official_split"]["source"])


def page_about():
    need_model()
    st.title("About the deployed model")
    st.markdown(f"""
- **Classifier:** {info['model_name']} — {info['training_method']}
- **Anomaly detector:** {info['anomaly_detector']}, threshold at a
  {info['if_target_false_alarm_rate']:.0%} target false-alarm rate on held-out normal rows
- **Alert rule:** alert if the classifier predicts any attack class **or** the Isolation Forest
  flags the record (verdict *Anomaly* when only the forest fires)
- **Training data:** {info['training_samples']:,} deduplicated NSL-KDD rows,
  {info['feature_count']} features, {info['class_count']} classes
- **Preprocessing:** {info['preprocessing']}
- **Severity / response:** Phase 6 policy (Low → Monitor, Medium → Alert, High → Investigate,
  Critical → Immediate Response)
- **Trained at:** {info['trained_at']}
""")
    st.caption("Offline research prototype: predictions are not connected to any live network "
               "and no traffic is blocked.")
    with st.expander("Classes"):
        st.write(", ".join(info["classes"]))


{"Predict": page_predict, "Batch CSV": page_batch, "History": page_history,
 "Model performance": page_performance, "About": page_about}[page]()
