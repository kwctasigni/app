# --- IMPORTS ---
import streamlit as st

# --- PAGE SETUP ---
about_page = st.Page(
    "views/Overview.py",
    title="Overview",
    icon=":material/account_circle:",
    default=True,
)

performance_page = st.Page(
    "views/Performance.py",
    title="Performance",
    icon=":material/bar_chart:",
)
tool_page = st.Page(
    "views/Tool.py",
    title="Tool",
    icon=":material/smart_toy:",
)

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [about_page],
        "Analysis": [performance_page, tool_page],
    }
)


# --- SHARED ON ALL PAGES ---
st.sidebar.markdown("By [Alxkon](https://www.linkedin.com/in/konstantinos-alexiou-543735188/) 📶")


# --- RUN NAVIGATION ---
pg.run()



