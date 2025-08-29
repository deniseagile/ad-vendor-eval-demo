# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Ad Vendor Evaluation Simulator", layout="wide")
st.title("📊 Ad Vendor Evaluation Simulator")
st.caption("Public demo with synthetic data — Snowflake + Streamlit portfolio project")

DATA_DIR = Path("data")

@st.cache_data
def load_csvs():
    health   = pd.read_csv(DATA_DIR/"health.csv")
    delivery = pd.read_csv(DATA_DIR/"delivery.csv")
    finance  = pd.read_csv(DATA_DIR/"finance.csv")
    identity = pd.read_csv(DATA_DIR/"identity.csv")
    budgets  = pd.read_csv(DATA_DIR/"budgets.csv")
    return health, delivery, finance, identity, budgets

health, delivery, finance, identity, budgets = load_csvs()

# ---------------- Sidebar controls ----------------
st.sidebar.header("Controls")
vendors_all = health["VENDOR_ID"].unique().tolist()
sel_vendors = st.sidebar.multiselect("Select vendors", vendors_all, default=vendors_all)

st.sidebar.markdown("### Health Score Weights")
w_match    = st.sidebar.slider("Match Rate", 0.0, 1.0, 0.30, 0.01)
w_overlap  = st.sidebar.slider("Overlap",    0.0, 1.0, 0.15, 0.01)
w_cpm      = st.sidebar.slider("CPM (lower is better)", 0.0, 1.0, 0.25, 0.01)
w_view     = st.sidebar.slider("Viewability", 0.0, 1.0, 0.20, 0.01)
w_disputes = st.sidebar.slider("Disputes (lower is better)", 0.0, 1.0, 0.10, 0.01)

weights = [w_match, w_overlap, w_cpm, w_view, w_disputes]
weights = [w/sum(weights) for w in weights]  # normalize

# ---------------- Recompute Health ----------------
dyn = identity.merge(health[["VENDOR_ID"]], on="VENDOR_ID", how="right").fillna(0)
dyn = dyn.merge(finance.groupby("VENDOR_ID", as_index=False)["DISPUTE_RATE"].mean(),
                on="VENDOR_ID", how="left").fillna(0)
dyn = dyn.merge(delivery.groupby("VENDOR_ID", as_index=False)["CPM"].mean(),
                on="VENDOR_ID", how="left").fillna(0)
dyn = dyn.merge(delivery.groupby("VENDOR_ID", as_index=False)["VIEWABILITY"].mean(),
                on="VENDOR_ID", how="left").fillna(0)

dyn["HEALTH_SCORE_DYNAMIC"] = (
    weights[0]*dyn["MATCH_RATE"] +
    weights[1]*dyn["OVERLAP"] +
    weights[2]*(1 - (dyn["CPM"]/15)) +     # target CPM = 15
    weights[3]*((dyn["VIEWABILITY"] - 0.5)/0.5) +
    weights[4]*(1 - dyn["DISPUTE_RATE"])
).clip(0,1)*100

dyn = dyn[dyn["VENDOR_ID"].isin(sel_vendors)].sort_values("HEALTH_SCORE_DYNAMIC", ascending=False)

# ---------------- KPIs ----------------
c1, c2, c3 = st.columns(3)
c1.metric("Top Vendor", dyn.iloc[0]["VENDOR_ID"] if not dyn.empty else "—")
c2.metric("Top Score", f"{dyn['HEALTH_SCORE_DYNAMIC'].max():.1f}" if not dyn.empty else "—")
c3.metric("Avg Score", f"{dyn['HEALTH_SCORE_DYNAMIC'].mean():.1f}" if not dyn.empty else "—")

# ---------------- Health Table ----------------
st.subheader("🏆 Partnership Health Ranking")
st.dataframe(dyn[["VENDOR_ID","HEALTH_SCORE_DYNAMIC"]].rename(
    columns={"VENDOR_ID":"Vendor","HEALTH_SCORE_DYNAMIC":"Health Score"}
), use_container_width=True)

# ---------------- Delivery Trends ----------------
st.subheader("📈 Delivery Trends")
dt_col = "DT"
delivery_sel = delivery[delivery["VENDOR_ID"].isin(sel_vendors)]
if not delivery_sel.empty:
    delivery_sel[dt_col] = pd.to_datetime(delivery_sel[dt_col])
    chart = alt.Chart(delivery_sel).mark_line().encode(
        x=dt_col+":T", y="CPM:Q", color="VENDOR_ID:N",
        tooltip=["DT","VENDOR_ID","CPM","CTR"]
    ).properties(height=350)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No delivery data for selected vendors.")

# ---------------- Finance ----------------
st.subheader("💰 Budget vs Invoiced Variance")
finance_summary = finance.merge(budgets, on=["MONTH","VENDOR_ID"], how="left")
finance_summary["VARIANCE_USD"] = finance_summary["INVOICED_USD"] - finance_summary["BUDGET_ALLOCATED_USD"]
st.dataframe(finance_summary[["MONTH","VENDOR_ID","BUDGET_ALLOCATED_USD","INVOICED_USD","VARIANCE_USD","DISPUTE_RATE"]],
             use_container_width=True)

# ---------------- Identity Scatter ----------------
st.subheader("🧩 Identity Match vs Overlap")
chart_id = alt.Chart(identity[identity["VENDOR_ID"].isin(sel_vendors)]).mark_circle(size=200).encode(
    x=alt.X("MATCH_RATE:Q", axis=alt.Axis(format="%")),
    y=alt.Y("OVERLAP:Q", axis=alt.Axis(format="%")),
    color="VENDOR_ID:N",
    tooltip=["VENDOR_ID","MATCH_RATE","OVERLAP","AVG_LATENCY_HOURS"]
).properties(height=350)
st.altair_chart(chart_id, use_container_width=True)

st.caption("Synthetic demo: health score combines Match Rate, Overlap, CPM, Viewability, and Disputes.")
