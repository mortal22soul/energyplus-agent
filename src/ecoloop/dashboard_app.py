"""Streamlit dashboard entry point for Eco-Loop.
Run with: streamlit run dashboard_app.py
"""
from __future__ import annotations

from ecoloop.config import RunConfig
from ecoloop.dashboard import render_dashboard

config = RunConfig()
render_dashboard(config)
