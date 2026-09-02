"""
app/dashboard.py

Streamlit dashboard for the Pearls AQI Predictor.
Communicates with the FastAPI backend via HTTP.

Run (after starting the backend):
    streamlit run app/dashboard.py
"""

import os
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

try:
    API_URL = st.secrets["AQI_API_URL"]
except Exception:
    API_URL = os.environ.get("AQI_API_URL", "http://localhost:8000")
TARGETS = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]
HORIZON_LABEL = {
    "AQI_t+1": "Tomorrow",
    "AQI_t+2": "In 2 Days",
    "AQI_t+3": "In 3 Days",
}
HORIZON_ICON = {
    "AQI_t+1": "\u23f0",
    "AQI_t+2": "\U0001f552",
    "AQI_t+3": "\U0001f553",
}
HORIZON_SHORT = {
    "AQI_t+1": "T+1",
    "AQI_t+2": "T+2",
    "AQI_t+3": "T+3",
}

ALERT_CATEGORIES = {
    "Unhealthy for Sensitive Groups": ("warning", "\u26a0\ufe0f"),
    "Unhealthy": ("warning", "\u26a0\ufe0f"),
    "Very Unhealthy": ("danger", "\U0001f6a8"),
    "Hazardous": ("danger", "\U0001f6a8"),
}

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Karachi AQI \u00b7 Air Quality Intelligence",
    page_icon="\U0001f3d8\ufe0f",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# GLOBAL STYLES
# ============================================================

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1400px;}
    body {font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;}

    .section-card {
        background: linear-gradient(145deg, rgba(30,32,44,0.95), rgba(22,24,33,0.98));
        border: 1px solid rgba(255,255,255,0.06); border-radius: 16px;
        padding: 24px; margin-bottom: 8px; box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    }

    .kpi-card {
        background: linear-gradient(145deg, rgba(30,32,44,0.9), rgba(22,24,33,0.95));
        border: 1px solid rgba(255,255,255,0.06); border-radius: 16px;
        padding: 28px 20px; text-align: center; position: relative; overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.4);}
    .kpi-card .kpi-icon {font-size: 1.6rem; margin-bottom: 8px; display: block; opacity: 0.9;}
    .kpi-card .kpi-horizon {
        font-size: 0.75rem; color: #6b7280; text-transform: uppercase;
        letter-spacing: 0.1em; font-weight: 600; margin-bottom: 12px;
    }
    .kpi-card .kpi-value {
        font-size: 2.8rem; font-weight: 800; line-height: 1;
        margin-bottom: 12px; letter-spacing: -0.02em;
    }
    .kpi-card .kpi-badge {
        display: inline-block; padding: 5px 14px; border-radius: 999px;
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
    }
    .kpi-card .kpi-accent {
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        border-radius: 16px 16px 0 0;
    }

    .kpi-card-sm {
        background: linear-gradient(145deg, rgba(30,32,44,0.9), rgba(22,24,33,0.95));
        border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
        padding: 18px 14px; text-align: center; box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    .kpi-card-sm .kpi-horizon {
        font-size: 0.65rem; color: #6b7280; text-transform: uppercase;
        letter-spacing: 0.1em; font-weight: 600; margin-bottom: 6px;
    }
    .kpi-card-sm .kpi-value {
        font-size: 1.8rem; font-weight: 800; line-height: 1; margin-bottom: 8px;
    }
    .kpi-card-sm .kpi-badge {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.6rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
    }

    .metric-pill {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 8px 16px; margin: 0 4px;
    }
    .metric-pill .metric-label {
        font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.06em;
    }
    .metric-pill .metric-value {font-size: 0.95rem; font-weight: 700; color: #e5e7eb;}

    .alert-banner {
        border-radius: 12px; padding: 14px 20px; margin-bottom: 16px;
        font-weight: 600; font-size: 0.9rem; display: flex; align-items: center; gap: 10px;
    }
    .alert-warning {
        background: linear-gradient(135deg, rgba(251,146,60,0.12), rgba(251,146,60,0.06));
        border: 1px solid rgba(251,146,60,0.35); color: #fdba74;
    }
    .alert-danger {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.06));
        border: 1px solid rgba(239,68,68,0.4); color: #fca5a5;
    }
    .alert-ok {
        background: linear-gradient(135deg, rgba(34,197,94,0.10), rgba(34,197,94,0.04));
        border: 1px solid rgba(34,197,94,0.3); color: #86efac;
    }

    .section-header {
        font-size: 0.75rem; color: #6b7280; text-transform: uppercase;
        letter-spacing: 0.12em; font-weight: 700; margin-bottom: 12px;
        padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    div[data-testid="stExpander"] {
        border-radius: 14px; border: 1px solid rgba(255,255,255,0.08);
        background: rgba(30,32,44,0.6);
    }
    div[data-testid="stExpander"] summary {font-weight: 600; color: #d1d5db;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {border-radius: 8px 8px 0 0; font-size: 0.85rem; font-weight: 600;}
    .stSelectbox > div > div {border-radius: 10px;}
    .stButton > button {border-radius: 10px; font-weight: 600; letter-spacing: 0.02em;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# API HELPERS
# ============================================================

@st.cache_data(ttl=60)
def api_get(path: str, params=None):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, json_body: dict):
    r = requests.post(f"{API_URL}{path}", json=json_body, timeout=30)
    r.raise_for_status()
    return r.json()


# ============================================================
# HEALTH CHECK
# ============================================================

try:
    api_get("/health")
except Exception:
    st.error("Dashboard can't reach the backend. Make sure it's running, then refresh.")
    st.stop()


# ============================================================
# HELPER: history stats
# ============================================================

def get_history_stats(hist_df):
    aqi = hist_df["AQI"].dropna()
    return {
        "current": float(aqi.iloc[-1]) if len(aqi) > 0 else None,
        "average": float(aqi.mean()) if len(aqi) > 0 else None,
        "max": float(aqi.max()) if len(aqi) > 0 else None,
        "min": float(aqi.min()) if len(aqi) > 0 else None,
    }


# ============================================================
# RENDER FUNCTIONS
# ============================================================

def render_alerts(predictions, context=""):
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
        prefix = f"{context} \u2014 " if context else ""
        css = "alert-danger" if worst_severity == "danger" else "alert-warning"
        st.markdown(
            f'<div class="alert-banner {css}">'
            f'\U0001f6a8 {prefix}Hazardous air quality predicted &mdash; '
            f'{" &middot; ".join(messages)}</div>',
            unsafe_allow_html=True,
        )
    else:
        if context:
            st.markdown(
                f'<div class="alert-banner alert-ok">'
                f'\u2705 {context} \u2014 All forecasted horizons are within safe levels</div>',
                unsafe_allow_html=True,
            )


def render_kpi_cards(predictions, size="large"):
    cols = st.columns(3, gap="medium")
    for col, target in zip(cols, TARGETS):
        p = predictions[target]
        color = p["color"]
        css = "kpi-card" if size == "large" else "kpi-card-sm"
        with col:
            st.markdown(
                f"""<div class="{css}">
                    <div class="kpi-accent" style="background: linear-gradient(90deg, {color}, transparent);"></div>
                    <span class="kpi-icon">{HORIZON_ICON[target]}</span>
                    <div class="kpi-horizon">{HORIZON_LABEL[target]}</div>
                    <div class="kpi-value" style="color: {color};">{p['predicted_aqi']}</div>
                    <span class="kpi-badge" style="background: {color}22; color: {color}; border: 1px solid {color}44;">
                        {p['category']}
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )


def render_history_chart(hist_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["AQI"], mode="lines",
        line=dict(color="#3b82f6", width=2, shape="spline", smoothing=0.8),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>AQI: <b>%{y:.0f}</b><extra></extra>",
        name="AQI",
    ))
    if len(hist_df) > 7:
        rolling = hist_df["AQI"].rolling(7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=hist_df["date"], y=rolling, mode="lines",
            line=dict(color="#a855f7", width=1.5, dash="dot"),
            hovertemplate="7-day avg: <b>%{y:.1f}</b><extra></extra>",
            name="7-day avg", visible="legendonly",
        ))
    fig.update_layout(
        height=320, margin=dict(t=20, b=30, l=10, r=10),
        xaxis=dict(showgrid=False, title=None, tickfont=dict(size=11, color="#6b7280"),
                    linecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title=dict(text="AQI", font=dict(size=12, color="#6b7280")),
                    showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                    tickfont=dict(size=11, color="#6b7280")),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af", family="Segoe UI, sans-serif"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(17,24,39,0.95)", bordercolor="rgba(255,255,255,0.1)",
                        font=dict(size=12, color="#f3f4f6")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10, color="#6b7280")),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_chart(data, value_col, color, title):
    fig = go.Figure(go.Bar(
        x=data[value_col], y=data["feature"], orientation="h",
        marker=dict(
            color=data[value_col].apply(
                lambda v: color if v > 0 else "#22c55e" if value_col == "shap_value" else color
            ),
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=440, margin=dict(t=10, b=30, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af", family="Segoe UI, sans-serif", size=11),
        xaxis=dict(title=dict(text=title, font=dict(size=12, color="#6b7280")),
                    showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                    zeroline=True, zerolinecolor="rgba(255,255,255,0.08)",
                    tickfont=dict(size=10, color="#6b7280")),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10, color="#9ca3af")),
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(168,85,247,0.08));
    border: 1px solid rgba(255,255,255,0.06); border-radius: 16px;
    padding: 20px 28px; margin-bottom: 20px;
    display: flex; justify-content: space-between; align-items: center;">
    <div>
        <div style="font-size: 1.6rem; font-weight: 800; color: #f3f4f6; letter-spacing: -0.02em;">
            \U0001f3d8\ufe0f Karachi Air Quality Intelligence
        </div>
        <div style="font-size: 0.8rem; color: #6b7280; margin-top: 4px; letter-spacing: 0.02em;">
            Real-time AQI forecasting \u00b7 Powered by XGBoost + SHAP \u00b7 3-day outlook
        </div>
    </div>
    <div style="text-align: right;">
        <div style="font-size: 0.65rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em;">Last updated</div>
        <div style="font-size: 0.85rem; color: #9ca3af; font-weight: 600;">""" + time.strftime("%b %d, %Y \u00b7 %H:%M") + """</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# CONTROLS + DATA
# ============================================================

ctrl1, ctrl2 = st.columns([5, 1])
with ctrl2:
    history_days = st.selectbox("Range", [30, 60, 90, 180, 365], index=2,
                                label_visibility="collapsed", key="history_range")

pred_resp = api_get("/predict/latest")
predictions = pred_resp["predictions"]

hist = api_get("/history", params={"days": history_days})
hist_df = pd.DataFrame(hist["data"])
hist_df["date"] = pd.to_datetime(hist_df["date"])
stats = get_history_stats(hist_df)

# ---- Alerts ----
render_alerts(predictions)

# ---- Summary metrics ----
m1, m2, m3, m4 = st.columns(4)
cur = stats["current"]
cur_color = "#3b82f6" if cur and cur <= 100 else "#ef4444" if cur and cur > 200 else "#f59e0b"
with m1:
    st.markdown(f"""<div class="metric-pill"><div><div class="metric-label">Current AQI</div>
    <div class="metric-value" style="color:{cur_color};">{cur:.0f}</div></div></div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-pill"><div><div class="metric-label">Avg ({history_days}d)</div>
    <div class="metric-value">{stats['average']:.0f}</div></div></div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-pill"><div><div class="metric-label">Peak</div>
    <div class="metric-value" style="color:#ef4444;">{stats['max']:.0f}</div></div></div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-pill"><div><div class="metric-label">Low</div>
    <div class="metric-value" style="color:#22c55e;">{stats['min']:.0f}</div></div></div>""", unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)


# ============================================================
# SECTION 1: AQI TREND
# ============================================================

st.markdown('<div class="section-header">\U0001f4c8  AQI Trend</div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
render_history_chart(hist_df)
st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# SECTION 2: 3-DAY FORECAST
# ============================================================

st.markdown('<div class="section-header">\U0001f52e  3-Day Forecast</div>', unsafe_allow_html=True)
render_kpi_cards(predictions, size="large")
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# ============================================================
# SECTION 3: WHAT-IF SIMULATOR
# ============================================================

with st.expander("\U0001f504  What-If Simulator \u2014 Adjust conditions to see impact on forecast", expanded=False):
    s1, s2, s3 = st.columns(3)
    with s1:
        temp = st.slider("Temperature (\u00b0C)", 10.0, 45.0, 28.0, key="wi_temp")
    with s2:
        pm25 = st.slider("PM2.5 (\u03bcg/m\u00b3)", 0.0, 200.0, 25.0, key="wi_pm25")
    with s3:
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, key="wi_humidity")
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        run_whatif = st.button("\u26a1  Run Simulation", type="primary", use_container_width=True)
    if run_whatif:
        overrides = {"Temperature": temp, "PM2.5": pm25, "Humidity": humidity}
        whatif = api_post("/predict", {"overrides": overrides})
        st.session_state["whatif_predictions"] = whatif["predictions"]
        st.session_state["last_overrides"] = overrides
    if "whatif_predictions" in st.session_state:
        st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
        render_alerts(st.session_state["whatif_predictions"], context="What-if scenario")
        st.markdown('<div class="section-header" style="border: none; padding-bottom: 0;">\U0001f4ca  Adjusted Forecast</div>', unsafe_allow_html=True)
        render_kpi_cards(st.session_state["whatif_predictions"], size="small")


# ============================================================
# ============================================================
# SECTION 4: HEALTH ADVISORY
# ============================================================

st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">\U0001f3e5  Health Advisory</div>', unsafe_allow_html=True)

try:
    advisory_resp = api_get("/health-advisory")
    advisories = advisory_resp["advisories"]

    # --- Per-horizon advisory cards ---
    adv_cols = st.columns(3, gap="medium")
    for col, target in zip(adv_cols, TARGETS):
        adv = advisories[target]
        color = adv["color"]
        with col:
            st.markdown(f'''
            <div class="kpi-card" style="padding: 22px 16px; text-align: left;">
                <div class="kpi-accent" style="background: linear-gradient(90deg, {color}, transparent);"></div>
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    <span style="font-size: 1.4rem;">{adv["icon"]}</span>
                    <div>
                        <div class="kpi-horizon" style="margin-bottom: 0;">{HORIZON_LABEL[target]}</div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: {color};">{adv["level"]}</div>
                    </div>
                </div>
                <div style="font-size: 0.8rem; color: #9ca3af; line-height: 1.5; margin-bottom: 12px;">
                    {adv["summary"]}
                </div>
            </div>
            ''', unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # --- Detailed advisory for Tomorrow (T+1) ---
    t1_adv = advisories["AQI_t+1"]
    st.markdown(f'<div class="section-header" style="border: none; padding-bottom: 0; font-size: 0.7rem;">'
                f'\U0001f4cb  Detailed Advisory for Tomorrow (AQI {t1_adv["predicted_aqi"]})</div>',
                unsafe_allow_html=True)

    g1, g2 = st.columns([3, 2])

    with g1:
        st.markdown("**\U0001f465  Vulnerable Groups**")
        group_labels = {
            "general": "\U0001f464  General Public",
            "children": "\U0001f476  Children",
            "elderly": "\U0001f9d3  Elderly (65+)",
            "respiratory": "\U0001fac0  Respiratory Conditions",
            "cardiac": "\u2764\ufe0f  Heart Conditions",
        }
        for key, label in group_labels.items():
            advice_text = t1_adv["groups"].get(key, "")
            st.markdown(
                f'''<div style="
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 10px;
                    padding: 10px 14px;
                    margin-bottom: 6px;
                    font-size: 0.82rem;
                    color: #d1d5db;
                    line-height: 1.4;
                ">
                    <span style="font-weight: 700;">{label}</span><br/>
                    <span style="color: #9ca3af;">{advice_text}</span>
                </div>''',
                unsafe_allow_html=True,
            )

    with g2:
        st.markdown("**\u26a1  Recommended Actions**")
        for action in t1_adv["actions"]:
            st.markdown(
                f'''<div style="
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 10px;
                    padding: 10px 14px;
                    margin-bottom: 6px;
                    font-size: 0.82rem;
                    color: #d1d5db;
                ">{action}</div>''',
                unsafe_allow_html=True,
            )

except Exception:
    st.info("Health advisory unavailable — make sure the backend is running.")


# SECTION 5: FEATURE IMPORTANCE (SHAP)
# ============================================================

st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">\U0001f50d  Feature Importance (SHAP)</div>', unsafe_allow_html=True)

col_sel, col_tab = st.columns([1, 3])
with col_sel:
    horizon = st.selectbox("Forecast horizon", TARGETS,
                           format_func=lambda t: HORIZON_LABEL[t], key="fi_horizon")
    st.markdown(f'<div class="metric-pill" style="margin-top: 4px;">'
                f'<div><div class="metric-label">Selected</div>'
                f'<div class="metric-value">{HORIZON_SHORT[horizon]}</div></div></div>',
                unsafe_allow_html=True)

tab_local, tab_global = st.tabs(["\U0001f3af  This Forecast", "\U0001f4ca  Global Overview"])

with tab_local:
    overrides = st.session_state.get("last_overrides")
    if overrides:
        explanation = api_post("/explain", {"target": horizon, "overrides": overrides})
    else:
        explanation = api_get("/explain/latest", params={"target": horizon})
    contrib_df = pd.DataFrame(explanation["contributions"]).head(12).sort_values("shap_value")
    render_feature_chart(contrib_df, "shap_value", "#3b82f6",
                         "SHAP value (impact on predicted AQI)")
    st.caption("\U0001f534 Positive \u2192 pushes AQI higher     \U0001f7e2 Negative \u2192 pushes AQI lower")

with tab_global:
    global_exp = api_get("/explain/global", params={"target": horizon, "sample_size": 200})
    imp_df = pd.DataFrame(global_exp["importance"]).head(12).sort_values("mean_abs_shap")
    render_feature_chart(imp_df, "mean_abs_shap", "#a855f7",
                         "Mean |SHAP value| (global importance)")
    st.caption(f"\U0001f4cb  Averaged over {global_exp['n_samples']} historical samples")


# ============================================================
# FOOTER
# ============================================================

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
    color: #4b5563; font-size: 0.72rem; letter-spacing: 0.02em;">
    Karachi AQI Predictor \u00b7 Pearls Project \u00b7 XGBoost + SHAP \u00b7 Streamlit + FastAPI
</div>
""", unsafe_allow_html=True)
