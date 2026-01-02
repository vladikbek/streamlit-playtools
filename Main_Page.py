import streamlit as st

# CRITICAL: set_page_config MUST be the first Streamlit command
st.set_page_config(
    layout="wide",
    page_title="Playlist Tools",
    page_icon=":material/motion_play:",
    initial_sidebar_state="collapsed",
)


search_playlists = st.Page(
    "pages/1_Search_Playlists.py",
    title="Search Playlists",
    icon=":material/search:",
    url_path="search_playlists",
)
check_playlist = st.Page(
    "pages/2_Check_Playlist.py",
    title="Check Playlist",
    icon=":material/playlist_add_check:",
    url_path="check_playlist",
)
generate_playlist = st.Page(
    "pages/3_Generate_Playlist.py",
    title="Generate Playlist",
    icon=":material/playlist_add:",
    url_path="generate_playlist",
)
search_by_label = st.Page(
    "pages/4_Search_by_Label.py",
    title="Search by Label",
    icon=":material/library_music:",
    url_path="search_by_label",
)
viral_tracks = st.Page(
    "pages/5_Viral_Tracks.py",
    title="Viral Tracks",
    icon=":material/chart_data:",
    url_path="viral_tracks",
)
playlist_seo = st.Page(
    "pages/6_Playlist_SEO.py",
    title="Playlist SEO",
    icon=":material/analytics:",
    url_path="playlist_seo",
)


def _make_redirect(page: st.Page):
    def _redirect():
        st.switch_page(page)

    return _redirect


router_page = st.Page(_make_redirect(search_playlists), title="Search Playlists", default=True)

legacy_search_playlists = st.Page(
    _make_redirect(search_playlists),
    title="Search Playlists",
    url_path="Search_Playlists",
)
legacy_check_playlist = st.Page(
    _make_redirect(check_playlist),
    title="Check Playlist",
    url_path="Check_Playlist",
)
legacy_generate_playlist = st.Page(
    _make_redirect(generate_playlist),
    title="Generate Playlist",
    url_path="Generate_Playlist",
)
legacy_search_by_label = st.Page(
    _make_redirect(search_by_label),
    title="Search by Label",
    url_path="Search_by_Label",
)
legacy_viral_tracks = st.Page(
    _make_redirect(viral_tracks),
    title="Viral Tracks",
    url_path="Viral_Tracks",
)
legacy_playlist_seo = st.Page(
    _make_redirect(playlist_seo),
    title="Playlist SEO",
    url_path="Playlist_SEO",
)

with st.sidebar:
    st.page_link(search_playlists, label="Search Playlists", icon=":material/search:")
    st.page_link(check_playlist, label="Check Playlist", icon=":material/playlist_add_check:")
    st.page_link(generate_playlist, label="Generate Playlist", icon=":material/playlist_add:")
    st.page_link(search_by_label, label="Search by Label", icon=":material/library_music:")
    st.page_link(viral_tracks, label="Viral Tracks", icon=":material/chart_data:")
    st.page_link(playlist_seo, label="Playlist SEO", icon=":material/analytics:")

page = st.navigation(
    [
        router_page,
        search_playlists,
        check_playlist,
        generate_playlist,
        search_by_label,
        viral_tracks,
        playlist_seo,
        legacy_search_playlists,
        legacy_check_playlist,
        legacy_generate_playlist,
        legacy_search_by_label,
        legacy_viral_tracks,
        legacy_playlist_seo,
    ],
    position="hidden",
)
page.run()
