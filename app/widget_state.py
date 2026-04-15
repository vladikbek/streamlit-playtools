import streamlit as st


def sync_widget_state(state_key: str, value) -> None:
    """Initialize widget state from query params without clobbering user edits on reruns."""
    if state_key not in st.session_state:
        st.session_state[state_key] = value
