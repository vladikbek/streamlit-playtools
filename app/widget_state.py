from copy import deepcopy

import streamlit as st


def sync_widget_state(state_key: str, value) -> None:
    """Keep widget state stable while still accepting external query-param updates."""
    snapshot_key = f"__query_snapshot__{state_key}"
    incoming_value = deepcopy(value)

    if state_key not in st.session_state:
        st.session_state[state_key] = incoming_value
        st.session_state[snapshot_key] = incoming_value
        return

    if snapshot_key not in st.session_state:
        st.session_state[snapshot_key] = incoming_value
        return

    if incoming_value != st.session_state[snapshot_key]:
        st.session_state[state_key] = incoming_value
        st.session_state[snapshot_key] = incoming_value
