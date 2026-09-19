import streamlit as st
import pandas as pd
import sys
import os

# Ensure analytics_db can be imported
sys.path.append(os.path.abspath("."))
from analytics_db import get_total_users, get_questions_today, get_avg_score, get_daily_requests, get_user_activity

st.set_page_config(page_title="Admin Analytics", layout="wide", page_icon="📊")

# --- Security Check ---
if "authentication_status" not in st.session_state or not st.session_state["authentication_status"]:
    st.error("You must be logged in as an Admin to view this page.")
    st.stop()

st.title("📊 Startup Admin Dashboard")
st.markdown("Monitor your SaaS usage, LLM performance, and active users in real-time.")

# Fetch Real Data
total_users = get_total_users()
questions_today = get_questions_today()
avg_score = get_avg_score()
mrr = total_users * 10  # Assuming $10/mo per user

# Metrics
st.markdown("### 📈 Real-Time Usage Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", f"{total_users}")
col2.metric("Questions Asked Today", f"{questions_today}")
col3.metric("Avg Judge Score (Faithfulness)", f"{avg_score:.2f}")
col4.metric("MRR (Monthly Recurring Revenue)", f"${mrr:,}")

st.markdown("---")
st.markdown("### 🧠 LLM Request Volume (Last 7 Days)")
daily_data = get_daily_requests()
if daily_data:
    df_daily = pd.DataFrame(daily_data, columns=["Date", "Queries"])
    df_daily.set_index("Date", inplace=True)
    st.bar_chart(df_daily)
else:
    st.info("No queries logged yet in the last 7 days.")

st.markdown("---")
st.markdown("### 👥 Active Subscriptions")
activity_data = get_user_activity()
if activity_data:
    df_activity = pd.DataFrame(activity_data, columns=["User", "Queries Today"])
    df_activity["Plan"] = "Premium ($10/mo)"
    df_activity.loc[df_activity["User"] == "admin", "Plan"] = "Admin"
    df_activity["Status"] = "Active"
    st.dataframe(df_activity, use_container_width=True)
else:
    st.info("No user activity logged yet.")
