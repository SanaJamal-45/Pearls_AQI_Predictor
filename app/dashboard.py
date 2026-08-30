"""
app/dashboard.py

Streamlit dashboard for the Pearls AQI Predictor.
Run (after starting the backend):

    streamlit run app/dashboard.py
"""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = st.secrets.get("AQI_API_URL", os.environ.get("AQI_API_URL", "http://localhost:8000"))
TARGETS = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]
HORIZON_LABEL = {"AQI_t+1": "Tomorrow", "AQI_t+2": "In 2 days", "AQI_t+3": "In 3 days"}

# Categories that trigger the hazardous-air alert banner
ALERT_CATEGORIES = {
    "Unhealthy for Sensitive Groups": ("warning", "⚠️"),
    "Unhealthy": ("warning", "⚠️"),
    "Very Unhealthy": ("danger", "🚨"),
    "Hazardous": ("danger", "🚨"),
}

st.set_page_config(page_title="Karachi AQI Dashboard", page_icon="🌫️", layout="wide")

st.markdown(
    """
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {padding-top: 2.2rem; max-width: 1200px;}

        /* ---- Forecast / what-if cards ---- */
        .aqi-card {
            border-radius: 16px;
            padding: 20px 16px;
            text-align: center;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        }
        .aqi-card .label {
            font-size: 0.8rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }
        .aqi-card .value {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 8px;
        }
        .aqi-card .badge {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
        }

        /* ---- Alert banner ---- */
        .alert-banner {
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 18px;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .alert-warning {
            background: rgba(255, 126, 0, 0.12);
            border: 1px solid rgba(255, 126, 0, 0.5);
            color: #ffb066;
        }
        .alert-danger {
            background: rgba(220, 38, 38, 0.15);
            border: 1px solid rgba(220, 38, 38, 0.6);
            color: #ff8080;
        }

        h1 {font-size: 1.7rem !important; margin-bottom: 0.2rem !important;}
        h3 {margin-top: 0.4rem;}

        div[data-testid="stExpander"] {
            border-radius: 14px;
            border: 1px solid #262b36;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def api_get(path: str, params: dict | None = None):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, json_body: dict):
    r = requests.post(f"{API_URL}{path}", json=json_body, timeout=30)
    r.raise_for_status()
    return r.json()


try:
    api_get("/health")
except Exception:
    st.error("Dashboard can't reach the backend. Make sure it's running, then refresh.")
    st.stop()


def render_cards(predictions: dict, size: str = "large"):
    """Renders the 3-horizon forecast cards. `size='small'` is used for
    the what-if section so it doesn't visually compete with the main
    forecast above it."""
    value_size = "2rem" if size == "large" else "1.4rem"
    cols = st.columns(3)
    for col, target in zip(cols, TARGETS):
        p = predictions[target]
        with col:
            st.markdown(
                f"""
                <div class="aqi-card" style="--card-bg:{p['color']}14; --card-border:{p['color']}44;">
                    <div class="label">{HORIZON_LABEL[target]}</div>
                    <div class="value" style="font-size:{value_size}; color:{p['color']};">{p['predicted_aqi']}</div>
                    <span class="badge" style="background:{p['color']}26; color:{p['color']};">{p['category']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_alerts(predictions: dict, context: str = ""):
    """Shows a banner for any horizon predicted to be Unhealthy or worse."""
    worst_severity = None
    messages = []
    for target in TARGETS:
        p = predictions[target]
        if p["category"] in ALERT_CATEGORIES:
            severity, icon = ALERT_CATEGORIES[p["category"]]
            messages.append(f"{icon} {HORIZON_LABEL[target]}: {p['predicted_aqi']} ({p['category']})")
            if severity == "danger":
                worst_severity = "danger"
            elif worst_severity != "danger":
                worst_severity = "warning"

    if worst_severity:
        prefix = f"{context} — " if context else ""
        css_class = "alert-danger" if worst_severity == "danger" else "alert-warning"
        st.markdown(
            f"""<div class="alert-banner {css_class}">
                {prefix}Hazardous air quality predicted: {' · '.join(messages)}
            </div>""",
            unsafe_allow_html=True,
        )


# ============================================================
# Header + controls
# ============================================================

st.title("Karachi AQI Dashboard")

col_title, col_range = st.columns([3, 1])
with col_range:
    history_days = st.selectbox("Range", [30, 90, 180, 365], index=1, label_visibility="collapsed")


# ============================================================
# Alerts (checked against the baseline forecast up front)
# ============================================================

pred_resp = api_get("/predict/latest")
predictions = pred_resp["predictions"]

render_alerts(predictions)


# ============================================================
# History chart
# ============================================================

hist = api_get("/history", params={"days": history_days})
hist_df = pd.DataFrame(hist["data"])
hist_df["date"] = pd.to_datetime(hist_df["date"])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_df["date"], y=hist_df["AQI"],
    mode="lines",
    line=dict(color="#60a5fa", width=2.5, shape="spline", smoothing=0.5),
    fill="tozeroy",
    fillcolor="rgba(96,165,250,0.12)",
    hovertemplate="%{x|%b %d}<br>AQI: %{y:.0f}<extra></extra>",
))
fig.update_layout(
    height=280,
    margin=dict(t=10, b=10, l=10, r=10),
    xaxis=dict(showgrid=False, title=None),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title=None, zeroline=False),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9ca3af"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Forecast cards
# ============================================================

render_cards(predictions, size="large")

st.write("")


# ============================================================
# What-if sliders
# ============================================================

with st.expander("Adjust conditions"):
    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.slider("Temperature (°C)", 10.0, 45.0, 28.0)
    with c2:
        pm25 = st.slider("PM2.5", 0.0, 200.0, 25.0)
    with c3:
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)

    overrides = {"Temperature": temp, "PM2.5": pm25, "Humidity": humidity}
    if st.button("Update forecast"):
        whatif = api_post("/predict", {"overrides": overrides})
        st.session_state["whatif_predictions"] = whatif["predictions"]
        st.session_state["last_overrides"] = overrides

    if "whatif_predictions" in st.session_state:
        st.write("")
        render_alerts(st.session_state["whatif_predictions"], context="What-if scenario")
        render_cards(st.session_state["whatif_predictions"], size="small")

st.write("")


# ============================================================
# Feature importance
# ============================================================

st.subheader("Feature Importance")

horizon = st.selectbox("", TARGETS, format_func=lambda t: HORIZON_LABEL[t], label_visibility="collapsed")
tab_local, tab_global = st.tabs(["This forecast", "Overall"])

with tab_local:
    overrides = st.session_state.get("last_overrides")
    if overrides:
        explanation = api_post("/explain", {"target": horizon, "overrides": overrides})
    else:
        explanation = api_get("/explain/latest", params={"target": horizon})

    contrib_df = pd.DataFrame(explanation["contributions"]).head(12).sort_values("shap_value")
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in contrib_df["shap_value"]]

    fig2 = go.Figure(go.Bar(
        x=contrib_df["shap_value"], y=contrib_df["feature"], orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    fig2.update_layout(
        height=420, margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af"),
        xaxis=dict(title="Impact on predicted AQI", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title=None),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🔴 increases predicted AQI · 🟢 decreases predicted AQI")

with tab_global:
    global_exp = api_get("/explain/global", params={"target": horizon, "sample_size": 100})
    imp_df = pd.DataFrame(global_exp["importance"]).head(12).sort_values("mean_abs_shap")

    fig3 = go.Figure(go.Bar(
        x=imp_df["mean_abs_shap"], y=imp_df["feature"], orientation="h",
        marker_color="#60a5fa",
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    fig3.update_layout(
        height=420, margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af"),
        xaxis=dict(title="Average impact", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title=None),
    )
    st.plotly_chart(fig3, use_container_width=True)