import streamlit as st

st.set_page_config(
    page_title="Top Songs Finder",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("🎵 Welcome to Top Songs Finder")

st.markdown("""
## About the App

Top Songs Finder helps you discover the most popular and trending songs across Spotify playlists. 
Our intelligent algorithm analyzes user-created playlists to find tracks that are:

- 🔥 Trending across multiple playlists
- 📈 Popular among listeners
- 🆕 Recently released
- 🎯 Relevant to your search

## How to Use

1. Go to the **Search Tracks** page
2. Enter keywords related to the type of music you're looking for
3. Adjust the settings in the sidebar if needed
4. Click Search to discover top tracks
5. View detailed analytics in the **Analytics** page

## Features

- Multi-market search support
- Customizable scoring weights
- Playlist filtering options
- Record label information
- Detailed analytics and insights

Get started by navigating to the **Search Tracks** page in the sidebar! 👈
""")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Spotify API") 