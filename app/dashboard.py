"""
app/dashboard.py

Karachi Pearls AQI Predictor — Streamlit dashboard.
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
HORIZON_LABEL = {"AQI_t+1": "Tomorrow", "AQI_t+2": "In 2 Days", "AQI_t+3": "In 3 Days"}
HORIZON_SHORT = {"AQI_t+1": "T+1", "AQI_t+2": "T+2", "AQI_t+3": "T+3"}

# ============================================================
# SVG ICONS — single-quoted to avoid conflicts with " inside SVGs
# ============================================================

ICO = {}

ICO["wind"] = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg>'
ICO["trend"] = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>'
ICO["cal"] = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/></svg>'
ICO["heart"] = '<svg width="16" height="16" viewBox="0 0 24 24" fill="#ef4444" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19.5 12.572l-7.5 7.428l-7.5-7.428A5 5 0 1 1 12 2.003a5 5 0 1 1 7.5 10.569"/></svg>'
ICO["search"] = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" x2="16.65" y1="21" y2="16.65"/></svg>'
ICO["c1"] = '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'
ICO["c2"] = '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 8 10"/></svg>'
ICO["c3"] = '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 10"/></svg>'
ICO["alert"] = '<svg width="16" height="16" viewBox="0 0 24 24" fill="#f59e0b" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>'
ICO["stop"] = '<svg width="16" height="16" viewBox="0 0 24 24" fill="#ef4444" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>'
ICO["check"] = '<svg width="14" height="14" viewBox="0 0 24 24" fill="#22c55e" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
ICO["shield"] = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
ICO["clip"] = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></svg>'
ICO["users"] = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
ICO["zap"] = '<svg width="14" height="14" viewBox="0 0 24 24" fill="#f59e0b" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'
ICO["person"] = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
ICO["baby"] = '<svg width="14" height="14" viewBox="0 0 24 24" fill="#f59e0b" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="5"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>'
ICO["senior"] = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="7" r="4"/><path d="M6 21v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2"/><path d="M10 11h4"/></svg>'
ICO["lungs"] = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6.081 20C3.8 20 2 18.2 2 16V9c0-1.1.9-2 2-2h1"/><path d="M17.919 20c2.2 0 4-1.8 4-4V9c0-1.1-.9-2-2-2h-1"/><path d="M12 4v16"/></svg>'
ICO["heart_m"] = '<svg width="14" height="14" viewBox="0 0 24 24" fill="#ef4444" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19.5 12.572l-7.5 7.428l-7.5-7.428A5 5 0 1 1 12 2.003a5 5 0 1 1 7.5 10.569"/></svg>'

ALERT_CATEGORIES = {
    "Unhealthy for Sensitive Groups": ("warning", ICO["alert"]),
    "Unhealthy": ("warning", ICO["alert"]),
    "Very Unhealthy": ("danger", ICO["stop"]),
    "Hazardous": ("danger", ICO["stop"]),
}

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Karachi Pearls AQI Predictor",
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
    .kpi-card .kpi-icon {margin-bottom: 8px; display: flex; justify-content: center;}
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
    .kpi-card-sm .kpi-value {font-size: 1.8rem; font-weight: 800; line-height: 1; margin-bottom: 8px;}
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
        display: flex; align-items: center; gap: 8px;
    }
    .adv-group {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;
        font-size: 0.82rem; color: #d1d5db; line-height: 1.4;
    }
    .adv-group .gl {font-weight: 700; display: flex; align-items: center; gap: 6px;}
    .adv-group .gt {color: #9ca3af; margin-top: 4px;}
    .act-card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;
        font-size: 0.82rem; color: #d1d5db; display: flex; align-items: center; gap: 8px;
    }
    div[data-testid="stExpander"] {
        border-radius: 14px; border: 1px solid rgba(255,255,255,0.08);
        background: rgba(30,32,44,0.6);
    }
    div[data-testid="stExpander"] summary {font-weight: 600; color: #d1d5db;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {border-radius: 8px 8px 0 0; font-size: 0.85rem; font-weight: 600;}
    .stButton > button {border-radius: 10px; font-weight: 600; letter-spacing: 0.02em;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# API HELPERS
# ============================================================

@st.cache_data(ttl=60)
def api_get(path, params=None):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=90)
    r.raise_for_status()
    return r.json()

def api_post(path, body):
    r = requests.post(f"{API_URL}{path}", json=body, timeout=90)
    r.raise_for_status()
    return r.json()

# Retry logic: Render free tier cold-starts take 30-60s.
# Try up to 3 times with increasing waits before giving up.
def _api_get_with_retry(path, params=None, retries=3, delay=15):
    last_exc = None
    for i in range(retries):
        try:
            r = requests.get(f"{API_URL}{path}", params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            if i < retries - 1:
                time.sleep(delay)
    raise last_exc

try:
    _api_get_with_retry("/health")
except Exception as e:
    st.error(
        f"Dashboard can't reach the backend at:\n\n"
        f"`{API_URL}`\n\n"
        f"Error: {e}\n\n"
        f"**Possible fixes:**\n"
        f"1. The Render backend may be waking up from sleep — wait 1 minute and refresh\n"
        f"2. Check your Streamlit secret `AQI_API_URL` is correct\n"
        f"3. Verify the backend is live at: {API_URL}/health"
    )
    st.stop()

def get_history_stats(df):
    aqi = df["AQI"].dropna()
    return {
        "current": float(aqi.iloc[-1]) if len(aqi) > 0 else None,
        "average": float(aqi.mean()) if len(aqi) > 0 else None,
        "max": float(aqi.max()) if len(aqi) > 0 else None,
        "min": float(aqi.min()) if len(aqi) > 0 else None,
    }


# ============================================================
# RENDER FUNCTIONS
# ============================================================

def render_alerts(preds, context=""):
    worst = None
    msgs = []
    for t in TARGETS:
        p = preds[t]
        if p["category"] in ALERT_CATEGORIES:
            sev, icon = ALERT_CATEGORIES[p["category"]]
            msgs.append(f"{HORIZON_LABEL[t]}: {p['predicted_aqi']} ({p['category']})")
            if sev == "danger": worst = "danger"
            elif worst != "danger": worst = "warning"
    if worst:
        prefix = f"{context} &mdash; " if context else ""
        css = "alert-danger" if worst == "danger" else "alert-warning"
        icon = ICO["stop"] if worst == "danger" else ICO["alert"]
        st.markdown(f'<div class="alert-banner {css}">{icon} {prefix}Hazardous air quality predicted &mdash; {" &middot; ".join(msgs)}</div>', unsafe_allow_html=True)
    elif context:
        st.markdown(f'<div class="alert-banner alert-ok">{ICO["check"]} {context} &mdash; All forecasted horizons are within safe levels</div>', unsafe_allow_html=True)


def render_kpi(preds, size="large"):
    css = "kpi-card" if size == "large" else "kpi-card-sm"
    vsize = "2.8rem" if size == "large" else "1.8rem"
    icons = {"AQI_t+1": ICO["c1"], "AQI_t+2": ICO["c2"], "AQI_t+3": ICO["c3"]}
    cols = st.columns(3, gap="medium")
    for col, t in zip(cols, TARGETS):
        p = preds[t]
        c = p["color"]
        with col:
            st.markdown(f'<div class="{css}"><div class="kpi-accent" style="background:linear-gradient(90deg,{c},transparent)"></div><div class="kpi-icon">{icons[t]}</div><div class="kpi-horizon">{HORIZON_LABEL[t]}</div><div class="kpi-value" style="color:{c}">{p["predicted_aqi"]}</div><span class="kpi-badge" style="background:{c}22;color:{c};border:1px solid {c}44">{p["category"]}</span></div>', unsafe_allow_html=True)


def render_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["AQI"], mode="lines",
        line=dict(color="#3b82f6", width=2, shape="spline", smoothing=0.8),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>AQI: <b>%{y:.0f}</b><extra></extra>", name="AQI"))
    if len(df) > 7:
        r = df["AQI"].rolling(7, center=True).mean()
        fig.add_trace(go.Scatter(x=df["date"], y=r, mode="lines",
            line=dict(color="#a855f7", width=1.5, dash="dot"),
            hovertemplate="7-day avg: <b>%{y:.1f}</b><extra></extra>", name="7-day avg", visible="legendonly"))
    fig.update_layout(height=320, margin=dict(t=20, b=30, l=10, r=10),
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#6b7280")),
        yaxis=dict(title=dict(text="AQI", font=dict(size=12, color="#6b7280")), showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(size=11, color="#6b7280")),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af", family="Segoe UI"), hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(17,24,39,0.95)", bordercolor="rgba(255,255,255,0.1)", font=dict(size=12, color="#f3f4f6")))
    st.plotly_chart(fig, use_container_width=True)


def render_feature(data, col, color, title):
    fig = go.Figure(go.Bar(x=data[col], y=data["feature"], orientation="h",
        marker=dict(color=data[col].apply(lambda v: color if v > 0 else "#22c55e" if col == "shap_value" else color), line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>"))
    fig.update_layout(height=440, margin=dict(t=10, b=30, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af", family="Segoe UI", size=11),
        xaxis=dict(title=dict(text=title, font=dict(size=12, color="#6b7280")), showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=True, zerolinecolor="rgba(255,255,255,0.08)", tickfont=dict(size=10, color="#6b7280")),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10, color="#9ca3af")), bargap=0.3)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(f"""<div style="background:linear-gradient(135deg,rgba(59,130,246,0.12),rgba(168,85,247,0.08));border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:20px 28px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center">
<div style="display:flex;align-items:center;gap:14px">
{ICO["wind"]}
<div>
<div style="font-size:1.5rem;font-weight:800;color:#f3f4f6;letter-spacing:-0.02em">Karachi Pearls AQI Predictor</div>
<div style="font-size:0.8rem;color:#6b7280;margin-top:4px;letter-spacing:0.02em">Real-time air quality forecasting &middot; XGBoost + SHAP &middot; 3-day outlook</div>
</div></div>
<div style="text-align:right">
<div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em">Last updated</div>
<div style="font-size:0.85rem;color:#9ca3af;font-weight:600">{time.strftime("%b %d, %Y &middot; %H:%M")}</div>
</div></div>""", unsafe_allow_html=True)


# ============================================================
# CONTROLS + DATA
# ============================================================

ctrl1, ctrl2 = st.columns([5, 1])
with ctrl2:
    history_days = st.selectbox("Range", [30, 60, 90, 180, 365], index=2, label_visibility="collapsed", key="hr")

pred_resp = api_get("/predict/latest")
predictions = pred_resp["predictions"]
hist = api_get("/history", params={"days": history_days})
hist_df = pd.DataFrame(hist["data"])
hist_df["date"] = pd.to_datetime(hist_df["date"])
stats = get_history_stats(hist_df)

render_alerts(predictions)

m1, m2, m3, m4 = st.columns(4)
cur = stats["current"]
cc = "#3b82f6" if cur and cur <= 100 else "#ef4444" if cur and cur > 200 else "#f59e0b"
with m1:
    st.markdown(f'<div class="metric-pill"><div><div class="metric-label">Current AQI</div><div class="metric-value" style="color:{cc}">{cur:.0f}</div></div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-pill"><div><div class="metric-label">Avg ({history_days}d)</div><div class="metric-value">{stats["average"]:.0f}</div></div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-pill"><div><div class="metric-label">Peak</div><div class="metric-value" style="color:#ef4444">{stats["max"]:.0f}</div></div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-pill"><div><div class="metric-label">Low</div><div class="metric-value" style="color:#22c55e">{stats["min"]:.0f}</div></div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# ============================================================
# SECTION 1: AQI TREND
# ============================================================

st.markdown(f'<div class="section-header">{ICO["trend"]} AQI Trend</div>', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
render_chart(hist_df)
st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# SECTION 2: 3-DAY FORECAST
# ============================================================

st.markdown(f'<div class="section-header">{ICO["cal"]} 3-Day Forecast</div>', unsafe_allow_html=True)
render_kpi(predictions, size="large")
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ============================================================
# SECTION 3: WHAT-IF SIMULATOR
# ============================================================

with st.expander("What-If Simulator \u2014 Adjust conditions to see impact on forecast", expanded=False):
    s1, s2, s3 = st.columns(3)
    with s1: temp = st.slider("Temperature (\u00b0C)", 10.0, 45.0, 28.0, key="wi_t")
    with s2: pm25 = st.slider("PM2.5 (\u03bcg/m\u00b3)", 0.0, 200.0, 25.0, key="wi_p")
    with s3: humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, key="wi_h")
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("Run Simulation", type="primary", use_container_width=True):
            whatif = api_post("/predict", {"overrides": {"Temperature": temp, "PM2.5": pm25, "Humidity": humidity}})
            st.session_state["wi_pred"] = whatif["predictions"]
            st.session_state["wi_over"] = {"Temperature": temp, "PM2.5": pm25, "Humidity": humidity}
    if "wi_pred" in st.session_state:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        render_alerts(st.session_state["wi_pred"], context="What-if scenario")
        st.markdown(f'<div class="section-header" style="border:none;padding-bottom:0">{ICO["trend"]} Adjusted Forecast</div>', unsafe_allow_html=True)
        render_kpi(st.session_state["wi_pred"], size="small")


# ============================================================
# SECTION 4: HEALTH ADVISORY
# ============================================================

st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-header">{ICO["heart"]} Health Advisory</div>', unsafe_allow_html=True)

try:
    adv_resp = api_get("/health-advisory")
    advs = adv_resp["advisories"]

    acols = st.columns(3, gap="medium")
    for col, t in zip(acols, TARGETS):
        a = advs[t]
        with col:
            st.markdown(f'<div class="kpi-card" style="padding:22px 16px;text-align:left"><div class="kpi-accent" style="background:linear-gradient(90deg,{a["color"]},transparent)"></div><div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">{ICO["shield"]}<div><div class="kpi-horizon" style="margin-bottom:0">{HORIZON_LABEL[t]}</div><div style="font-size:1.1rem;font-weight:800;color:{a["color"]}">{a["level"]}</div></div></div><div style="font-size:0.8rem;color:#9ca3af;line-height:1.5">{a["summary"]}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    t1 = advs["AQI_t+1"]
    st.markdown(f'<div class="section-header" style="border:none;padding-bottom:0;font-size:0.7rem">{ICO["clip"]} Detailed Advisory for Tomorrow (AQI {t1["predicted_aqi"]})</div>', unsafe_allow_html=True)

    g1, g2 = st.columns([3, 2])

    with g1:
        st.markdown(f"**{ICO['users']}  Vulnerable Groups**", unsafe_allow_html=True)
        gl = {"general": (ICO["person"], "General Public"), "children": (ICO["baby"], "Children"), "elderly": (ICO["senior"], "Elderly (65+)"), "respiratory": (ICO["lungs"], "Respiratory Conditions"), "cardiac": (ICO["heart_m"], "Heart Conditions")}
        for k, (ic, lbl) in gl.items():
            st.markdown(f'<div class="adv-group"><div class="gl">{ic} {lbl}</div><div class="gt">{t1["groups"].get(k,"")}</div></div>', unsafe_allow_html=True)

    with g2:
        st.markdown(f"**{ICO['zap']}  Recommended Actions**", unsafe_allow_html=True)
        for act in t1["actions"]:
            st.markdown(f'<div class="act-card">{ICO["check"]} {act}</div>', unsafe_allow_html=True)

except Exception:
    st.info("Health advisory unavailable \u2014 make sure the backend is running.")


# ============================================================
# SECTION 5: FEATURE IMPORTANCE (SHAP)
# ============================================================

st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-header">{ICO["search"]} Feature Importance (SHAP)</div>', unsafe_allow_html=True)

col_sel, col_tab = st.columns([1, 3])
with col_sel:
    horizon = st.selectbox("Forecast horizon", TARGETS, format_func=lambda t: HORIZON_LABEL[t], key="fi_h")
    st.markdown(f'<div class="metric-pill" style="margin-top:4px"><div><div class="metric-label">Selected</div><div class="metric-value">{HORIZON_SHORT[horizon]}</div></div></div>', unsafe_allow_html=True)

tab_l, tab_g = st.tabs(["This Forecast", "Global Overview"])

with tab_l:
    over = st.session_state.get("wi_over")
    if over:
        exp = api_post("/explain", {"target": horizon, "overrides": over})
    else:
        exp = api_get("/explain/latest", params={"target": horizon})
    cdf = pd.DataFrame(exp["contributions"]).head(12).sort_values("shap_value")
    render_feature(cdf, "shap_value", "#3b82f6", "SHAP value (impact on predicted AQI)")
    st.caption("Positive \u2192 pushes AQI higher     Negative \u2192 pushes AQI lower")

with tab_g:
    gexp = api_get("/explain/global", params={"target": horizon, "sample_size": 200})
    idf = pd.DataFrame(gexp["importance"]).head(12).sort_values("mean_abs_shap")
    render_feature(idf, "mean_abs_shap", "#a855f7", "Mean |SHAP value| (global importance)")
    st.caption(f"Averaged over {gexp['n_samples']} historical samples")


# ============================================================
# FOOTER
# ============================================================

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
st.markdown(f"""<div style="text-align:center;padding:16px;border-top:1px solid rgba(255,255,255,0.06);color:#4b5563;font-size:0.72rem;letter-spacing:0.02em;display:flex;justify-content:center;align-items:center;gap:8px">
{ICO["wind"]} Karachi Pearls AQI Predictor &middot; XGBoost + SHAP &middot; Streamlit + FastAPI
</div>""", unsafe_allow_html=True)
