import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────────
train = pd.read_csv("data_train.csv")
oot   = pd.read_csv("data_oot.csv")

def fmt_gbp(v):
    if v >= 1_000_000: return f"£{v/1_000_000:.1f}m"
    if v >= 1_000:     return f"£{v/1_000:.1f}k"
    return f"£{v:.0f}"

def compute_metrics(df, flagged):
    y = df["is_fraud"].astype(int)
    flagged = flagged.astype(int)
    tp = int((flagged.astype(bool) & y.astype(bool)).sum())
    fp = int((flagged.astype(bool) & ~y.astype(bool)).sum())
    fn = int((~flagged.astype(bool) & y.astype(bool)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    amt_caught      = float(df.loc[flagged.astype(bool) & y.astype(bool), "transaction_amount"].sum())
    amt_total_fraud = float(df.loc[y.astype(bool), "transaction_amount"].sum())
    value_recall    = amt_caught / amt_total_fraud if amt_total_fraud > 0 else 0
    return dict(
        precision=precision, recall=recall, f1=f1,
        flag_rate=flagged.mean(), tp=tp, fp=fp,
        n_flagged=int(flagged.sum()), total_fraud=int(y.sum()),
        amt_caught=amt_caught, amt_total_fraud=amt_total_fraud, value_recall=value_recall,
    )

app = dash.Dash(__name__, title="Fraud Rule Explorer", suppress_callback_exceptions=True)
server = app.server

DARK  = "#0a0d14"
CARD  = "#131720"
CARD2 = "#1c2030"
TEAL  = "#00c9a7"
BLUE  = "#3d8ef8"
RED   = "#ff4d6d"
AMBER = "#ffaa00"
MUTED = "#6b7494"
WHITE = "#e8eaf6"
GREEN = "#36d399"
FONT  = "'Inter', 'Helvetica Neue', sans-serif"
MONO  = "'IBM Plex Mono', monospace"

METRIC_TIPS = {
    "Precision":      "Precision: of all flagged transactions, what % are genuine fraud. High precision = fewer false alarms.",
    "Recall":         "Recall: of all actual fraud cases, what % were caught by count. High recall = fewer missed frauds.",
    "F1 Score":       "F1: harmonic mean of precision and recall. Balances both when they trade off.",
    "Flag Rate":      "Flag Rate: share of all transactions flagged by the active rules.",
    "Fraud Caught":   "Fraud Caught (TP): number of fraud cases the active rules successfully identified.",
    "False Alerts":   "False Alerts (FP): legitimate transactions incorrectly flagged as fraud.",
    "Value Caught %": "Value Recall: £ value of fraud caught ÷ total £ value of all fraud in the dataset.",
    "Value Caught £": "Value Caught: absolute £ amount of fraud intercepted vs total fraudulent amount.",
}

DEFAULTS  = {"txn": 4, "spend": 400, "amount": 500}
SUGGESTED = {"txn": 4, "spend": 400, "amount": 500, "rf": 0.65}
SUGGESTED_REASONING = [
    ("Txn count ≥ 4 (24h)", "Catches velocity fraud without flagging heavy legitimate users (avg ~3 txns/day)."),
    ("Amount ≥ £500", "Targets high-value single transactions — elevated fraud rate above this threshold."),
    ("Late night ecom ON", "10–100× fraud rate between 0–4am on ecommerce; near-zero false positive cost."),
    ("Logic: OR", "Each rule independently strong enough — AND would miss too many cases."),
]

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
.metric-card-wrap {
    position: relative;
    flex: 1;
    min-width: 110px;
    display: flex;
}
.metric-tooltip {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: #2a3050;
    color: #e8eaf6;
    font-size: 12px;
    font-family: 'Inter', 'Helvetica Neue', sans-serif;
    line-height: 1.5;
    padding: 8px 12px;
    border-radius: 8px;
    white-space: nowrap;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s ease;
    z-index: 9999;
    border: 1px solid #3d8ef8;
}
.metric-card-wrap:hover .metric-tooltip { opacity: 1; }
.metric-inner {
    background: #1c2030;
    border-radius: 12px;
    padding: 16px 18px;
    width: 100%;
    cursor: help;
}
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>"""

def section_label(text, color=TEAL):
    return html.P(text, style={
        "color": color, "fontSize": "9px", "letterSpacing": "2.5px",
        "fontWeight": "600", "margin": "0 0 10px 0", "fontFamily": FONT
    })

def rule_block(toggle_id, toggle_label, slider_id, slider_label,
               min_, max_, step, value, active=True):
    return html.Div([
        html.Div([
            dcc.Checklist(id=toggle_id, options=[{"label": "", "value": "on"}],
                          value=["on"] if active else [],
                          style={"display": "inline-flex", "alignItems": "center"}),
            html.Span(toggle_label, style={"color": WHITE, "fontSize": "12px",
                      "fontWeight": "500", "marginLeft": "8px", "fontFamily": FONT})
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        html.Div([
            html.Span(slider_label, style={"color": MUTED, "fontSize": "11px", "fontFamily": FONT}),
            html.Span(id=f"{slider_id}-val", style={"color": TEAL, "fontSize": "11px",
                      "fontFamily": MONO, "fontWeight": "600"})
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
        dcc.Slider(id=slider_id, min=min_, max=max_, step=step, value=value,
                   marks={min_: str(min_), max_: str(max_)},
                   tooltip={"always_visible": False}, updatemode="drag")
    ], style={"background": CARD2, "borderRadius": "10px", "padding": "12px 14px", "marginBottom": "10px"})

def metric_card(label, value_id, color=WHITE, small=False):
    tip = METRIC_TIPS.get(label, "")
    return html.Div([
        html.Div([
            html.P(label, style={"color": MUTED, "fontSize": "10px", "letterSpacing": "1.5px",
                   "textTransform": "uppercase", "margin": "0 0 4px 0", "fontFamily": FONT}),
            html.H2(id=value_id, style={"color": color, "margin": 0,
                    "fontSize": "18px" if small else "24px",
                    "fontWeight": "700", "fontFamily": MONO}),
        ], className="metric-inner", style={"borderLeft": f"3px solid {color}"}),
        html.Div(tip, className="metric-tooltip"),
    ], className="metric-card-wrap")


app.layout = html.Div(style={
    "background": DARK, "minHeight": "100vh",
    "fontFamily": FONT, "color": WHITE, "padding": "20px 24px"
}, children=[

    html.Link(rel="stylesheet",
              href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap"),

    # ── Info banner ──
    html.Div([
        html.Span("Prototype fraud rule exploration tool · built by ",
                  style={"color": MUTED, "fontSize": "11px"}),
        html.A("Alicja Ulejczyk", href="https://www.linkedin.com/in/alicja-ulejczyk-ba4313146/",
               target="_blank",
               style={"color": TEAL, "fontSize": "11px", "textDecoration": "none",
                      "fontWeight": "600"}),
        html.Span(" · Hover over any metric card to see its definition · ",
                  style={"color": MUTED, "fontSize": "11px"}),
        html.A("Report issues or questions", href="https://www.linkedin.com/in/alicja-ulejczyk-ba4313146/",
               target="_blank",
               style={"color": BLUE, "fontSize": "11px", "textDecoration": "none"}),
    ], style={
        "background": CARD2, "borderRadius": "10px", "padding": "8px 16px",
        "marginBottom": "16px", "display": "flex", "alignItems": "center",
        "flexWrap": "wrap", "gap": "4px",
        "borderLeft": f"3px solid {TEAL}",
    }),

    # ── Header ──
    html.Div([
        html.Div([
            html.H1("Fraud Rule Explorer", style={
                "margin": "4px 0 2px 0", "fontSize": "22px", "fontWeight": "700",
                "letterSpacing": "-0.3px"
            }),
            html.P("Adjust rule thresholds and observe the precision–recall tradeoff in real time",
                   style={"color": MUTED, "margin": 0, "fontSize": "13px"})
        ]),
        html.Div([
            html.Span("Dataset: ", style={"color": MUTED, "fontSize": "12px"}),
            dcc.RadioItems(
                id="dataset",
                options=[
                    {"label": "Training  (Jan–Oct)", "value": "train"},
                    {"label": "OOT Holdout  (Nov–Dec)", "value": "oot"}
                ],
                value="oot", inline=True,
                labelStyle={"marginLeft": "16px", "fontSize": "12px", "color": WHITE}
            )
        ], style={"display": "flex", "alignItems": "center",
                  "background": CARD2, "borderRadius": "10px", "padding": "10px 16px"})
    ], style={"display": "flex", "justifyContent": "space-between",
              "alignItems": "flex-end", "marginBottom": "20px"}),

    # ── Main layout ──
    html.Div(style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}, children=[

        # ── Sidebar ──
        html.Div(style={"background": CARD, "borderRadius": "14px", "padding": "20px",
                        "width": "270px", "flexShrink": "0"}, children=[

            html.Button("✦ Apply suggested thresholds", id="suggest-btn", n_clicks=0,
                style={"width": "100%", "padding": "10px", "marginBottom": "16px",
                       "background": "linear-gradient(135deg, #1a2540, #1e2d50)",
                       "border": f"1px solid {TEAL}", "borderRadius": "10px",
                       "color": TEAL, "fontSize": "12px", "fontWeight": "600",
                       "fontFamily": FONT, "cursor": "pointer", "letterSpacing": "0.5px"}),

            html.Div(id="suggest-panel", style={"display": "none"}, children=[
                html.Div([
                    html.P("WHY THESE THRESHOLDS", style={"color": TEAL, "fontSize": "9px",
                           "letterSpacing": "2px", "fontWeight": "600",
                           "margin": "0 0 10px 0", "fontFamily": FONT}),
                    *[html.Div([
                        html.Span(rule, style={"color": WHITE, "fontSize": "11px",
                                  "fontWeight": "600", "fontFamily": FONT,
                                  "display": "block", "marginBottom": "2px"}),
                        html.Span(reason, style={"color": MUTED, "fontSize": "10px",
                                  "fontFamily": FONT, "lineHeight": "1.4", "display": "block"}),
                    ], style={"marginBottom": "10px"})
                    for rule, reason in SUGGESTED_REASONING],
                ], style={"background": CARD2, "borderRadius": "10px", "padding": "14px",
                          "marginBottom": "16px", "borderLeft": f"2px solid {TEAL}"})
            ]),

            section_label("VELOCITY RULES"),
            rule_block("txn-toggle", "Transaction count (24h)", "txn-slider", "Threshold:",
                       1, 20, 1, DEFAULTS["txn"]),
            rule_block("spend-toggle", "Total spend (24h)", "spend-slider", "Threshold: £",
                       0, 2000, 50, DEFAULTS["spend"], active=False),

            html.Hr(style={"borderColor": CARD2, "margin": "14px 0"}),
            section_label("AMOUNT RULE"),
            rule_block("amount-toggle", "Single transaction amount", "amount-slider", "Threshold: £",
                       0, 5000, 100, DEFAULTS["amount"]),

            html.Hr(style={"borderColor": CARD2, "margin": "14px 0"}),
            section_label("CHANNEL / TIME RULE"),
            html.Div([
                html.Div([
                    dcc.Checklist(id="late-night-toggle", options=[{"label": "", "value": "on"}],
                                  value=["on"], style={"display": "inline-flex"}),
                    html.Span("Late night ecom (0–4am)", style={"color": WHITE, "fontSize": "12px",
                              "fontWeight": "500", "marginLeft": "8px"})
                ], style={"display": "flex", "alignItems": "center"}),
                html.P("Flags ecommerce transactions between midnight and 4am — fraud rate 10–100× baseline",
                       style={"color": MUTED, "fontSize": "10px", "margin": "6px 0 0 0"})
            ], style={"background": CARD2, "borderRadius": "10px",
                      "padding": "12px 14px", "marginBottom": "10px"}),

            html.Hr(style={"borderColor": CARD2, "margin": "14px 0"}),
            section_label("ML MODEL", color=BLUE),
            html.Div([
                html.P("⚠ Prototype only — evaluate on OOT holdout, not training data",
                       style={"color": AMBER, "fontSize": "10px", "margin": "0",
                              "lineHeight": "1.4", "fontFamily": FONT,
                              "borderLeft": f"2px solid {AMBER}", "paddingLeft": "8px"})
            ], style={"background": CARD2, "borderRadius": "8px", "padding": "8px 10px",
                      "marginBottom": "10px"}),
            html.Div([
                html.Div([
                    dcc.Checklist(id="rf-toggle", options=[{"label": "", "value": "on"}],
                                  value=[], style={"display": "inline-flex"}),
                    html.Span("Random Forest score", style={"color": WHITE, "fontSize": "12px",
                              "fontWeight": "500", "marginLeft": "8px"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
                html.Div([
                    html.Span("Threshold:", style={"color": MUTED, "fontSize": "11px"}),
                    html.Span(id="rf-slider-val", style={"color": BLUE, "fontSize": "11px",
                              "fontFamily": MONO, "fontWeight": "600"})
                ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
                dcc.Slider(id="rf-slider", min=0.0, max=1.0, step=0.05, value=0.5,
                           marks={0: "0", 0.5: "0.5", 1: "1"},
                           tooltip={"always_visible": False}, updatemode="drag"),
                html.P("Higher threshold = more precise, fewer alerts",
                       style={"color": MUTED, "fontSize": "10px", "margin": "6px 0 0 0"})
            ], style={"background": CARD2, "borderRadius": "10px",
                      "padding": "12px 14px", "marginBottom": "10px"}),

            html.Hr(style={"borderColor": CARD2, "margin": "14px 0"}),
            section_label("COMBINATION LOGIC", color=AMBER),
            html.Div([
                dcc.RadioItems(id="logic",
                    options=[
                        {"label": "OR — flag if ANY rule fires (higher recall)", "value": "or"},
                        {"label": "AND — flag if ALL rules fire (higher precision)", "value": "and"}
                    ],
                    value="or",
                    labelStyle={"display": "block", "marginBottom": "10px",
                                "fontSize": "11px", "color": WHITE}),
                html.P(id="logic-explanation", style={"color": AMBER, "fontSize": "10px",
                       "margin": "4px 0 0 0", "fontStyle": "italic"})
            ], style={"background": CARD2, "borderRadius": "10px", "padding": "12px 14px"})
        ]),

        # ── Main content ──
        html.Div(style={"flex": "1", "minWidth": "0"}, children=[

            html.Div(style={"display": "flex", "gap": "10px", "marginBottom": "10px",
                            "flexWrap": "wrap"}, children=[
                metric_card("Precision",    "m-precision", TEAL),
                metric_card("Recall",       "m-recall",    BLUE),
                metric_card("F1 Score",     "m-f1",        WHITE),
                metric_card("Flag Rate",    "m-flagrate",  RED),
                metric_card("Fraud Caught", "m-caught",    AMBER),
                metric_card("False Alerts", "m-fp",        MUTED),
            ]),

            html.Div(style={"display": "flex", "gap": "10px", "marginBottom": "16px",
                            "flexWrap": "wrap"}, children=[
                metric_card("Value Caught %", "m-value-pct", GREEN),
                metric_card("Value Caught £", "m-value-gbp", GREEN),
            ]),

            html.Div(style={"display": "flex", "gap": "14px", "marginBottom": "14px"}, children=[
                html.Div([dcc.Graph(id="pr-chart", config={"displayModeBar": False})],
                         style={"background": CARD, "borderRadius": "14px",
                                "padding": "14px", "flex": "1"}),
                html.Div([dcc.Graph(id="bar-chart", config={"displayModeBar": False})],
                         style={"background": CARD, "borderRadius": "14px",
                                "padding": "14px", "flex": "1"}),
            ]),

            html.Div([dcc.Graph(id="breakdown-chart", config={"displayModeBar": False})],
                     style={"background": CARD, "borderRadius": "14px", "padding": "14px"}),
        ])
    ])
])


@callback(
    Output("txn-slider", "value"), Output("spend-slider", "value"),
    Output("amount-slider", "value"), Output("rf-slider", "value"),
    Output("rf-toggle", "value"), Output("spend-toggle", "value"),
    Output("late-night-toggle", "value"),
    Output("logic", "value"), Output("suggest-panel", "style"),
    Input("suggest-btn", "n_clicks"),
    State("suggest-panel", "style"),
    prevent_initial_call=True,
)
def apply_suggestions(n_clicks, panel_style):
    currently_visible = panel_style.get("display") != "none"
    new_style = {"display": "none"} if currently_visible else {"display": "block"}
    return (SUGGESTED["txn"], SUGGESTED["spend"], SUGGESTED["amount"], SUGGESTED["rf"],
            [], [], ["on"], "or", new_style)


@callback(
    Output("txn-slider-val", "children"), Output("spend-slider-val", "children"),
    Output("amount-slider-val", "children"), Output("rf-slider-val", "children"),
    Input("txn-slider", "value"), Input("spend-slider", "value"),
    Input("amount-slider", "value"), Input("rf-slider", "value"),
)
def update_labels(txn, spend, amt, rf):
    return f"≥ {int(txn)} txns", f"≥ £{int(spend)}", f"≥ £{int(amt)}", f"≥ {rf:.2f}"


@callback(Output("logic-explanation", "children"), Input("logic", "value"))
def update_logic_text(logic):
    return ("Cast a wide net — good for catching more fraud, more false alerts" if logic == "or"
            else "Surgical precision — fewer alerts, only highest-confidence fraud")


@callback(
    Output("m-precision", "children"), Output("m-recall", "children"),
    Output("m-f1", "children"), Output("m-flagrate", "children"),
    Output("m-caught", "children"), Output("m-fp", "children"),
    Output("m-value-pct", "children"), Output("m-value-gbp", "children"),
    Output("pr-chart", "figure"), Output("bar-chart", "figure"),
    Output("breakdown-chart", "figure"),
    Input("dataset", "value"),
    Input("txn-toggle", "value"), Input("txn-slider", "value"),
    Input("spend-toggle", "value"), Input("spend-slider", "value"),
    Input("amount-toggle", "value"), Input("amount-slider", "value"),
    Input("late-night-toggle", "value"),
    Input("rf-toggle", "value"), Input("rf-slider", "value"),
    Input("logic", "value"),
)
def update_all(dataset, txn_on, txn, spend_on, spend, amt_on, amt,
               late_night, rf_on, rf_thresh, logic):
    df = train.copy() if dataset == "train" else oot.copy()

    r_txn   = (df["txn_count_24h"] >= txn).astype(int)     if txn_on    else pd.Series(0, index=df.index)
    r_spend = (df["spend_24h"] >= spend).astype(int)        if spend_on  else pd.Series(0, index=df.index)
    r_amt   = (df["transaction_amount"] >= amt).astype(int) if amt_on    else pd.Series(0, index=df.index)
    r_late  = ((df["hour"].isin([0,1,2,3,4])) & (df["pos_entry_mode"] == "ecom")).astype(int) \
              if late_night else pd.Series(0, index=df.index)
    r_rf    = (df["rf_score"] >= rf_thresh).astype(int)     if rf_on     else pd.Series(0, index=df.index)

    active_rules = []
    if txn_on:     active_rules.append(r_txn)
    if spend_on:   active_rules.append(r_spend)
    if amt_on:     active_rules.append(r_amt)
    if late_night: active_rules.append(r_late)
    if rf_on:      active_rules.append(r_rf)

    if not active_rules:
        flagged = pd.Series(0, index=df.index)
    elif logic == "or":
        flagged = active_rules[0].copy()
        for r in active_rules[1:]: flagged |= r
    else:
        flagged = active_rules[0].copy()
        for r in active_rules[1:]: flagged &= r

    m = compute_metrics(df, flagged)
    total = len(df)

    # PR curve
    thresholds = np.linspace(0.01, 0.99, 60)
    pr_pts = []
    for t in thresholds:
        f = (df["rf_score"] >= t).astype(int)
        mm = compute_metrics(df, f)
        pr_pts.append((mm["recall"], mm["precision"], t))
    pr_df = pd.DataFrame(pr_pts, columns=["recall", "precision", "threshold"])

    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(
        x=pr_df["recall"], y=pr_df["precision"], mode="lines", name="RF Model",
        line=dict(color=BLUE, width=2.5),
        hovertemplate="RF threshold: %{customdata:.2f}<br>Recall: %{x:.1%}<br>Precision: %{y:.1%}",
        customdata=pr_df["threshold"]
    ))
    pr_fig.add_trace(go.Scatter(
        x=[m["recall"]], y=[m["precision"]], mode="markers", name="Active rules",
        marker=dict(color=RED, size=14, symbol="star", line=dict(color=WHITE, width=1)),
        hovertemplate=f"Active rules<br>Recall: {m['recall']:.1%}<br>Precision: {m['precision']:.1%}"
    ))
    pr_fig.update_layout(
        title=dict(text="Precision–Recall Curve  ·  RF model vs active rules",
                   font=dict(size=12, color=WHITE, family=FONT)),
        xaxis=dict(title="Recall", color=MUTED, gridcolor=CARD2,
                   tickformat=".0%", range=[0,1], fixedrange=True),
        yaxis=dict(title="Precision", color=MUTED, gridcolor=CARD2,
                   tickformat=".0%", range=[0,1], fixedrange=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family=FONT),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
        margin=dict(l=10, r=10, t=40, b=10), height=290, uirevision="pr-fixed",
    )

    # Per-rule bar chart
    rule_labels = [
        f"Txn count ≥{int(txn)}" if txn_on else "Txn count (off)",
        f"Spend ≥£{int(spend)}"  if spend_on else "Spend (off)",
        f"Amount ≥£{int(amt)}"   if amt_on else "Amount (off)",
        "Late night ecom"        if late_night else "Late night (off)",
        f"RF ≥{rf_thresh:.2f}"  if rf_on else "RF (off)",
    ]
    recalls, precs = [], []
    for rf_ in [r_txn, r_spend, r_amt, r_late, r_rf]:
        mm = compute_metrics(df, rf_)
        recalls.append(mm["recall"])
        precs.append(mm["precision"])

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(name="Recall", x=rule_labels, y=recalls,
                             marker_color=BLUE, opacity=0.85))
    bar_fig.add_trace(go.Bar(name="Precision", x=rule_labels, y=precs,
                             marker_color=TEAL, opacity=0.85))
    bar_fig.update_layout(
        title=dict(text="Individual Rule Performance", font=dict(size=12, color=WHITE, family=FONT)),
        barmode="group",
        xaxis=dict(color=MUTED, gridcolor=CARD2, tickangle=-15,
                   tickfont=dict(size=10), fixedrange=True),
        yaxis=dict(color=MUTED, gridcolor=CARD2, tickformat=".0%",
                   title="Rate", range=[0,1], fixedrange=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family=FONT, size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=40, b=10), height=290, uirevision="bar-fixed",
    )

    # Breakdown
    fn = m["total_fraud"] - m["tp"]
    tn = total - m["n_flagged"] - fn
    breakdown_fig = go.Figure(go.Bar(
        x=["Fraud caught\n(True Positives)", "False alerts\n(False Positives)",
           "Missed fraud\n(False Negatives)", "Correct clears\n(True Negatives)"],
        y=[m["tp"], m["fp"], fn, tn],
        marker_color=[TEAL, RED, AMBER, CARD2],
        text=[f"{m['tp']:,}", f"{m['fp']:,}", f"{fn:,}", f"{tn:,}"],
        textposition="outside", textfont=dict(color=WHITE, size=11)
    ))
    breakdown_fig.update_layout(
        title=dict(text="Transaction Outcome Breakdown",
                   font=dict(size=12, color=WHITE, family=FONT)),
        xaxis=dict(color=MUTED, fixedrange=True),
        yaxis=dict(color=MUTED, gridcolor=CARD2, title="# Transactions",
                   range=[0, int(total * 1.05)], fixedrange=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family=FONT),
        margin=dict(l=10, r=10, t=40, b=40), height=270, uirevision="breakdown-fixed",
    )

    return (
        f"{m['precision']:.1%}", f"{m['recall']:.1%}", f"{m['f1']:.3f}",
        f"{m['flag_rate']:.2%}", f"{m['tp']:,} / {m['total_fraud']:,}", f"{m['fp']:,}",
        f"{m['value_recall']:.1%}",
        f"{fmt_gbp(m['amt_caught'])} / {fmt_gbp(m['amt_total_fraud'])}",
        pr_fig, bar_fig, breakdown_fig
    )


if __name__ == "__main__":
    app.run(debug=True)
