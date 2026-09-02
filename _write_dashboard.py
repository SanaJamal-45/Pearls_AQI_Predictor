#!/usr/bin/env python3
"""Write dashboard with SVG icons and new heading."""
import pathlib

DASHBOARD_PATH = pathlib.Path("app/dashboard.py")

# Read the existing dashboard as a template
content = DASHBOARD_PATH.read_text(encoding="utf-8")

# 1. Change heading
content = content.replace(
    "Karachi Air Quality Intelligence",
    "Karachi Pearls AQI Predictor"
)

# 2. Replace emoji icons with SVG inline icons in section headers
svg_trend = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>'
svg_calendar = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/></svg>'
svg_refresh = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/></svg>'
svg_heart = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M19.5 12.572l-7.5 7.428l-7.5-7.428A5 5 0 1 1 12 2.003a5 5 0 1 1 7.5 10.569"/></svg>'
svg_search = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><circle cx="11" cy="11" r="8"/><line x1="21" x2="16.65" y1="21" y2="16.65"/></svg>'
svg_wind = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg>'
svg_shield = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
svg_clipboard = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></svg>'
svg_users = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
svg_zap = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'
svg_check = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
svg_alert = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>'
svg_octagon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>'
svg_clock1 = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'
svg_clock2 = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 8 10"/></svg>'
svg_clock3 = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 10"/></svg>'
svg_person = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
svg_baby = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="5"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>'
svg_senior = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="7" r="4"/><path d="M6 21v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2"/><path d="M10 11h4"/></svg>'
svg_lungs = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6.081 20C3.8 20 2 18.2 2 16V9c0-1.1.9-2 2-2h1"/><path d="M17.919 20c2.2 0 4-1.8 4-4V9c0-1.1-.9-2-2-2h-1"/><path d="M12 4v16"/></svg>'
svg_heart_med = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19.5 12.572l-7.5 7.428l-7.5-7.428A5 5 0 1 1 12 2.003a5 5 0 1 1 7.5 10.569"/></svg>'

# Replace section headers
content = content.replace("'\U0001f4c8  AQI Trend'", f"'{svg_trend} AQI Trend'")
content = content.replace("'\U0001f52e  3-Day Forecast'", f"'{svg_calendar} 3-Day Forecast'")
content = content.replace("'\U0001f3e5  Health Advisory'", f"'{svg_heart} Health Advisory'")
content = content.replace("'\U0001f50d  Feature Importance (SHAP)'", f"'{svg_search} Feature Importance (SHAP)'")

# Replace What-if expander title
content = content.replace(
    "'\U0001f504  What-If Simulator",
    "'What-If Simulator"
)

# Replace Adjusted Forecast sub-header
content = content.replace(
    "'\U0001f4ca  Adjusted Forecast",
    f"'{svg_trend} Adjusted Forecast"
)

# Replace Detailed Advisory sub-header
content = content.replace(
    "'\U0001f4cb  Detailed Advisory",
    f"'{svg_clipboard} Detailed Advisory"
)

# Replace Vulnerable Groups header
content = content.replace(
    "'**\U0001f465  Vulnerable Groups**'",
    f"\"**{svg_users}  Vulnerable Groups**\""
)

# Replace Recommended Actions header
content = content.replace(
    "'**\u26a1  Recommended Actions**'",
    f"\"**{svg_zap}  Recommended Actions**\""
)

# Replace health group labels
group_replacements = {
    "\U0001f464  General Public": f"{svg_person} General Public",
    "\U0001f476  Children": f"{svg_baby} Children",
    "\U0001f9d3  Elderly (65+)": f"{svg_senior} Elderly (65+)",
    "\U0001fac0  Respiratory Conditions": f"{svg_lungs} Respiratory Conditions",
    "\u2764\ufe0f  Heart Conditions": f"{svg_heart_med} Heart Conditions",
}
for old, new in group_replacements.items():
    content = content.replace(old, new)

# Replace alert icons
content = content.replace("'\U0001f6a8 {prefix}", f"'{svg_octagon} {prefix}")
content = content.replace("'\u2705 {context}", f"'{svg_check} {context}")

# Replace KPI card icon t
