import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import concurrent.futures
from app.config import BATCH_SIZE, MAX_WORKERS
from app.label_search import (
    parse_release_date,
    process_releases_batch,
    process_tracks_batch,
    search_releases_by_label,
    search_tracks_by_label,
)
from app.widget_state import sync_widget_state

# Configuration variables
def get_query_param(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value

def setup_spotify():
    """Initialize Spotify client with credentials from .env file"""
    load_dotenv()
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    return spotipy.Spotify(auth_manager=auth_manager)

# Page configuration
st.title(":material/library_music: Search by Label", anchor=False)
st.caption("Find tracks or releases by record label and explore popularity, release dates, and identifiers.")

# Initialize Spotify client
sp = setup_spotify()

current_year = datetime.now().year
year_options = ["All years"] + [str(current_year)] + [str(current_year - i) for i in range(1, 6)]
year_options = list(dict.fromkeys(year_options))
param_label = get_query_param("keyword") or ""
param_search_releases = get_query_param("mode") == "releases"

sync_widget_state("label_search_input", param_label)
sync_widget_state("label_search_mode", param_search_releases)

with st.container(border=True):
    with st.form("label_search_form", border=False):
        search_col1, search_col2 = st.columns([8, 3])

        with search_col1:
            st.text_input(
                "Enter record label name:",
                help="Enter the exact name of the record label",
                label_visibility="collapsed",
                placeholder="Paste label name",
                icon=":material/link:",
                key="label_search_input"
            )

        with search_col2:
            search_button = st.form_submit_button(
                "Search",
                type="primary",
                icon=":material/search:",
                width="stretch"
            )

        filter_col1, filter_col2 = st.columns([8, 3])

        with filter_col1:
            selected_year = st.selectbox(
                "Year",
                options=year_options,
                index=0,
                help="Filter by release year",
                label_visibility="collapsed"
            )

        with filter_col2:
            st.checkbox(
                "Search for Releases",
                help="Search Spotify releases instead of tracks for this label.",
                width="stretch",
                key="label_search_mode"
            )

    search_label = st.session_state["label_search_input"].strip()
    search_releases = st.session_state["label_search_mode"]

    if search_releases:
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            min_release_date = st.date_input(
                "Minimum Release Date",
                value=None,
                help="Filter releases after this date (leave empty for no filter)"
            )

        with filter_col2:
            min_popularity = st.slider(
                "Minimum Popularity",
                min_value=0,
                max_value=100,
                value=0,
                help="Filter releases by minimum popularity score"
            )

        isrc_filter = ""
    else:
        with st.container(horizontal=True, vertical_alignment="center"):
            min_release_date = st.date_input(
                "Minimum Release Date",
                value=None,
                help="Filter tracks released after this date (leave empty for no filter)"
            )

            min_popularity = st.slider(
                "Minimum Popularity",
                min_value=0,
                max_value=100,
                value=0,
                help="Filter tracks by minimum popularity score"
            )

            isrc_filter = st.text_input(
                "Filter by ISRC",
                value="",
                help="Enter one or more ISRCs (comma-separated). Will match any track containing the entered text.",
                placeholder="ISRC contains..."
            )

current_search_key = {
    "keyword": search_label,
    "mode": "releases" if search_releases else "tracks"
}
last_search_label = st.session_state.get("label_search_key")
auto_search = bool(param_label)
should_run = bool(search_label) and (
    search_button or (auto_search and current_search_key != last_search_label)
)

if should_run:
    search_year = None if selected_year == "All years" else int(selected_year)
    search_target = "releases" if search_releases else "tracks"

    with st.spinner(f"Searching {search_target} from {search_label}..."):
        if search_releases:
            items = search_releases_by_label(sp, search_label, year=search_year)
        else:
            items = search_tracks_by_label(sp, search_label, year=search_year)

        if not items:
            st.warning(f"No {search_target} found for label: {search_label}")
            st.stop()

        batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
        processed_items = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            processor = process_releases_batch if search_releases else process_tracks_batch
            futures = [executor.submit(processor, (sp, batch)) for batch in batches]

            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    processed_items.extend(batch_results)
                except Exception as e:
                    st.warning(f"Error processing some tracks: {str(e)}")
                    continue

        if not processed_items:
            st.warning(f"No {search_target} found matching the exact label: {search_label}")
            st.stop()

        df = pd.DataFrame(processed_items)
        if search_releases:
            df = df.drop_duplicates(subset=["id"])
        df['release_date'] = df['release_date'].apply(parse_release_date)
        if not search_releases:
            df['Stats'] = df['url'].apply(lambda x: f"https://www.mystreamcount.com/track/{x.split(':')[-1]}")

        df = df.sort_values(['popularity', 'release_date'], ascending=[False, False])

        if df.empty:
            st.error(f"No {search_target} found.")
            st.stop()

        st.session_state["label_results"] = {
            "df": df,
            "label": search_label,
            "year": search_year,
            "search_mode": "releases" if search_releases else "tracks"
        }
        st.session_state["label_search_key"] = current_search_key

        if search_button:
            st.query_params.from_dict({
                "keyword": search_label,
                "mode": "releases" if search_releases else "tracks"
            })

label_results = st.session_state.get("label_results")
if label_results:
    df = label_results["df"].copy()
    search_mode = label_results.get("search_mode", "tracks")
    is_release_search = search_mode == "releases"

    if min_popularity > 0:
        df = df[df['popularity'] >= min_popularity]

    if min_release_date:
        df = df[df['release_date'] >= pd.Timestamp(min_release_date)]

    if not is_release_search and isrc_filter:
        isrc_list = [isrc.strip().upper() for isrc in isrc_filter.split(',') if isrc.strip()]
        if isrc_list:
            isrc_mask = df['isrc'].str.contains('|'.join(isrc_list), case=False, na=False)
            df = df[isrc_mask]

    if df.empty:
        result_label = "releases" if is_release_search else "tracks"
        st.warning(f"No {result_label} found after applying the filters.")
        st.stop()

    year_label = "All years" if label_results["year"] is None else str(label_results["year"])
    result_label = "releases" if is_release_search else "tracks"
    st.subheader(f"Found {len(df)} {result_label}", anchor=False)
    st.caption(f"{label_results['label']} · {year_label}")

    primary_color = st.get_option("theme.primaryColor")
    result_tab_label = "Releases" if is_release_search else "Tracks"
    tracks_tab, analytics_tab, artists_tab = st.tabs([result_tab_label, "Analytics", "Artists"])

    if is_release_search:
        column_config = {
            "url": st.column_config.LinkColumn(
                "Link",
                display_text=":material/open_in_new:",
                help="Click to open the release in Spotify desktop/mobile app",
                width="small"
            ),
            "artwork_url": st.column_config.ImageColumn(
                "Artwork",
                width="small"
            ),
            "name": st.column_config.TextColumn(
                "Release Name",
                width="medium"
            ),
            "artists": st.column_config.ListColumn(
                "Artists",
                width="medium"
            ),
            "label": st.column_config.TextColumn(
                "Label",
                width="medium"
            ),
            "album_type": st.column_config.TextColumn(
                "Type",
                width="small"
            ),
            "total_tracks": st.column_config.NumberColumn(
                "Tracks",
                format="%d",
                width="small"
            ),
            "popularity": st.column_config.NumberColumn(
                "Popularity",
                format="%d",
                width="small"
            ),
            "release_date": st.column_config.DateColumn(
                "Release Date",
                width="small"
            ),
            "upc": st.column_config.TextColumn(
                "UPC",
                width="medium"
            ),
        }
        column_order = [
            "url",
            "artwork_url",
            "name",
            "artists",
            "label",
            "album_type",
            "total_tracks",
            "popularity",
            "release_date",
            "upc",
        ]
    else:
        column_config = {
            "url": st.column_config.LinkColumn(
                "Link",
                display_text=":material/open_in_new:",
                help="Click to open in Spotify desktop/mobile app",
                width="small"
            ),
            "artwork_url": st.column_config.ImageColumn(
                "Artwork",
                width="small"
            ),
            "name": st.column_config.TextColumn(
                "Track Name",
                width="medium"
            ),
            "artists": st.column_config.ListColumn(
                "Artists",
                width="medium"
            ),
            "popularity": st.column_config.NumberColumn(
                "Popularity",
                format="%d",
                width="small"
            ),
            "release_date": st.column_config.DateColumn(
                "Release Date",
                width="small"
            ),
            "isrc": st.column_config.TextColumn(
                "ISRC",
                width="small"
            ),
            "Stats": st.column_config.LinkColumn(
                "Stats",
                display_text=":material/query_stats:",
                width="small"
            )
        }
        column_order = [
            "url",
            "artwork_url",
            "name",
            "artists",
            "popularity",
            "release_date",
            "isrc",
            "Stats"
        ]

    with tracks_tab:
        df.index = range(1, len(df) + 1)
        st.dataframe(
            df,
            width="stretch",
            height=500,
            column_order=column_order,
            column_config=column_config,
            hide_index=False
        )

    # Calculate artist statistics for analytics
    artist_popularity = []
    for _, row in df.iterrows():
        for artist in row['artists']:
            artist_popularity.append({
                'artist': artist,
                'popularity': row['popularity']
            })

    artist_df = pd.DataFrame(artist_popularity)
    if not artist_df.empty:
        artist_stats = artist_df.groupby('artist').agg(
            avg_popularity=('popularity', 'mean'),
            track_count=('popularity', 'count')
        ).reset_index()
    else:
        artist_stats = pd.DataFrame(columns=['artist', 'avg_popularity', 'track_count'])

    with analytics_tab:
        non_zero_popularity = df[df['popularity'] > 0]['popularity']
        avg_popularity = non_zero_popularity.mean() if not non_zero_popularity.empty else 0
        metric_label = "Average Release Popularity" if is_release_search else "Average Track Popularity"
        unique_artist_count = artist_stats["artist"].nunique()
        returning_artist_count = int((artist_stats["track_count"] >= 3).sum()) if not artist_stats.empty else 0
        returning_artist_share = (
            (returning_artist_count / unique_artist_count) * 100
            if unique_artist_count
            else 0
        )

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Returning Artists (3+)", f"{returning_artist_share:.1f}%")
        with metric_col2:
            st.metric(metric_label, f"{avg_popularity:.1f}")

        col1, col2 = st.columns(2)

        with col1:
            fig_popularity = px.histogram(
                df,
                x="popularity",
                title=f"Distribution of {'Release' if is_release_search else 'Track'} Popularity",
                nbins=10
            )
            if primary_color:
                fig_popularity.update_traces(marker_color=primary_color)
            fig_popularity.update_layout(
                xaxis_title="Popularity Score",
                yaxis_title=f"Number of {'Releases' if is_release_search else 'Tracks'}"
            )
            st.plotly_chart(fig_popularity, width="stretch")

        with col2:
            if not artist_stats.empty:
                top_by_tracks = artist_stats.nlargest(10, 'track_count')
                fig_tracks = px.bar(
                    top_by_tracks,
                    x='artist',
                    y='track_count',
                    title=f"Top 10 Artists by Number of {'Releases' if is_release_search else 'Tracks'}",
                    labels={'artist': 'Artist', 'track_count': f"Number of {'Releases' if is_release_search else 'Tracks'}"}
                )
                if primary_color:
                    fig_tracks.update_traces(marker_color=primary_color)
                fig_tracks.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_tracks, width="stretch")
            else:
                st.info("No artist data available.")

        if not artist_stats.empty:
            top_by_popularity = artist_stats[artist_stats['track_count'] >= 2].nlargest(10, 'avg_popularity')
            fig_artist_popularity = px.bar(
                top_by_popularity,
                x='artist',
                y='avg_popularity',
                title="Top 10 Artists by Popularity",
                labels={'artist': 'Artist', 'avg_popularity': 'Average Popularity'}
            )
            if primary_color:
                fig_artist_popularity.update_traces(marker_color=primary_color)
            fig_artist_popularity.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_artist_popularity, width="stretch")
        else:
            st.info("No artist data available for popularity ranking.")

    with artists_tab:
        if artist_stats.empty:
            st.info("No artist statistics available.")
        else:
            artist_stats['avg_popularity'] = artist_stats['avg_popularity'].round(1)
            st.dataframe(
                artist_stats.sort_values('avg_popularity', ascending=False),
                column_config={
                    'artist': st.column_config.TextColumn("Artist"),
                    'avg_popularity': st.column_config.NumberColumn("Average Popularity", format="%.1f"),
                    'track_count': st.column_config.NumberColumn("Number of Tracks")
                },
                hide_index=True,
                width="stretch"
            )
