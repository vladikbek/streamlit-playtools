import streamlit as st

st.set_page_config(
    page_title="PlayTools",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("🎵 Welcome to PlayTools")

st.markdown("""
### PlayTools helps you discover the most popular playlists and trending songs across Spotify.           
""")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Spotify API") 