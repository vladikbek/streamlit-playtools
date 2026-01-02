import streamlit as st


st.set_page_config(layout="wide", initial_sidebar_state="auto")


def home_page() -> None:
    st.title(":material/motion_play: Welcome to Playlist Tools")
    st.markdown(
        """
### Playlist Tools helps you discover the most popular playlists and trending songs across Spotify.
"""
    )
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and Spotify API")


pages = [
    st.Page(home_page, title="Playlist Tools", icon=":material/motion_play:", default=True),
    st.Page(
        "pages/1_Search_Playlists.py",
        title="Search Playlists",
        icon=":material/search:",
        url_path="Search_Playlists",
    ),
    st.Page(
        "pages/2_Check_Playlist.py",
        title="Check Playlist",
        icon=":material/playlist_add_check:",
        url_path="Check_Playlist",
    ),
    st.Page(
        "pages/3_Generate_Playlist.py",
        title="Generate Playlist",
        icon=":material/playlist_add:",
        url_path="Generate_Playlist",
    ),
    st.Page(
        "pages/4_Search_by_Label.py",
        title="Search by Label",
        icon=":material/library_music:",
        url_path="Search_by_Label",
    ),
    st.Page(
        "pages/5_Viral_Tracks.py",
        title="Viral Tracks",
        icon=":material/chart_data:",
        url_path="Viral_Tracks",
    ),
    st.Page(
        "pages/6_Playlist_SEO.py",
        title="Playlist SEO",
        icon=":material/analytics:",
        url_path="Playlist_SEO",
    ),
]

current_page = st.navigation(pages, position="sidebar")
current_page.run()
