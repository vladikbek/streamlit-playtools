import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import concurrent.futures
from app.config import BATCH_SIZE, MAX_WORKERS, MAX_RESULTS

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

def search_tracks_by_label(sp: spotipy.Spotify, label: str, limit: int = 50, year: int | None = None) -> list:
    """Search for tracks by record label"""
    tracks = []
    offset = 0
    
    # Use exact match with quotes for more precise results
    query = f'label:"{label}"'
    if year:
        query += f' year:{year}'
    
    while True:
        results = sp.search(q=query, type='track', limit=limit, offset=offset)
        if not results['tracks']['items']:
            break
            
        tracks.extend(results['tracks']['items'])
        offset += limit
        
        # Stop if we've reached the maximum number of tracks
        if offset >= MAX_RESULTS:
            break
            
    return tracks

def process_tracks_batch(args):
    """Process a batch of tracks"""
    sp, tracks_batch = args
    processed_tracks = []
    
    for track in tracks_batch:
        try:
            # Safely get artist names, skipping any None values
            artists = []
            for artist in track.get('artists', []):
                if artist and isinstance(artist, dict) and artist.get('name'):
                    artists.append(artist['name'])
            
            # Skip track if no valid artists found
            if not artists:
                continue
            
            # Get album images safely
            images = track.get('album', {}).get('images', [])
            artwork_url = images[-1]['url'] if images else ''
            
            # Get ISRC from external_ids
            isrc = track.get('external_ids', {}).get('isrc', 'N/A')
            
            track_data = {
                'artwork_url': artwork_url,
                'name': track.get('name', 'Unknown Track'),
                'artists': artists,
                'isrc': isrc,
                'popularity': track.get('popularity', 0),
                'release_date': track.get('album', {}).get('release_date', ''),
                'url': track.get('uri', '')
            }
            processed_tracks.append(track_data)
        except Exception as e:
            continue  # Skip tracks that cause errors
            
    return processed_tracks

def parse_release_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    try:
        date_str = str(date_str)
        if len(date_str) == 4:
            return pd.to_datetime(f"{date_str}-01-01")
        if len(date_str) == 7:
            return pd.to_datetime(f"{date_str}-01")
        return pd.to_datetime(date_str)
    except Exception:
        return pd.NaT


# Page configuration
st.title(":material/library_music: Search by Label", anchor=False)
st.caption("Find tracks by record label and explore popularity, release dates, and ISRCs.")

# Initialize Spotify client
sp = setup_spotify()

current_year = datetime.now().year
year_options = ["All years"] + [str(current_year)] + [str(current_year - i) for i in range(1, 6)]
year_options = list(dict.fromkeys(year_options))
param_label = get_query_param("keyword") or ""

with st.container(border=True):
    with st.form("label_search_form", border=False):
        search_col1, search_col2 = st.columns([8, 3])

        with search_col1:
            search_label = st.text_input(
                "Enter record label name:",
                param_label,
                help="Enter the exact name of the record label",
                label_visibility="collapsed",
                placeholder="Paste label name",
                icon=":material/link:"
            )

        with search_col2:
            search_button = st.form_submit_button(
                "Search",
                type="primary",
                icon=":material/search:",
                width="stretch"
            )

        with st.container(horizontal=True, vertical_alignment="center"):
            selected_year = st.selectbox(
                "Year",
                options=year_options,
                index=0,
                help="Filter by release year",
                label_visibility="collapsed"
            )

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

last_search_label = st.session_state.get("label_search_key")
auto_search = bool(param_label)
should_run = bool(search_label) and (
    search_button or (auto_search and search_label != last_search_label)
)

if should_run:
    search_year = None if selected_year == "All years" else int(selected_year)

    with st.spinner(f"Searching tracks from {search_label}..."):
        items = search_tracks_by_label(sp, search_label, year=search_year)

        if not items:
            st.warning(f"No tracks found for label: {search_label}")
            st.stop()

        # Process items in parallel
        batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
        processed_items = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_tracks_batch, (sp, batch)) for batch in batches]

            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    processed_items.extend(batch_results)
                except Exception as e:
                    st.warning(f"Error processing some tracks: {str(e)}")
                    continue

        if not processed_items:
            st.warning(f"No tracks found matching the exact label: {search_label}")
            st.stop()

        # Convert to DataFrame
        df = pd.DataFrame(processed_items)
        df['release_date'] = df['release_date'].apply(parse_release_date)
        df['Stats'] = df['url'].apply(lambda x: f"https://www.mystreamcount.com/track/{x.split(':')[-1]}")

        # Sort by popularity
        df = df.sort_values('popularity', ascending=False)

        if df.empty:
            st.error("No tracks found.")
            st.stop()

        st.session_state["label_results"] = {
            "df": df,
            "label": search_label,
            "year": search_year
        }
        st.session_state["label_search_key"] = search_label

        if search_button:
            st.query_params.from_dict({
                "keyword": search_label
            })

label_results = st.session_state.get("label_results")
if label_results:
    df = label_results["df"].copy()

    if min_popularity > 0:
        df = df[df['popularity'] >= min_popularity]

    if min_release_date:
        df = df[df['release_date'] >= pd.Timestamp(min_release_date)]

    if isrc_filter:
        isrc_list = [isrc.strip().upper() for isrc in isrc_filter.split(',') if isrc.strip()]
        if isrc_list:
            isrc_mask = df['isrc'].str.contains('|'.join(isrc_list), case=False, na=False)
            df = df[isrc_mask]

    if df.empty:
        st.warning("No tracks found after applying the filters.")
        st.stop()

    year_label = "All years" if label_results["year"] is None else str(label_results["year"])
    st.subheader(f"Found {len(df)} tracks", anchor=False)
    st.caption(f"{label_results['label']} · {year_label}")

    primary_color = st.get_option("theme.primaryColor")
    tracks_tab, analytics_tab, artists_tab = st.tabs(["Tracks", "Analytics", "Artists"])

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

    with tracks_tab:
        df.index = range(1, len(df) + 1)
        st.dataframe(
            df,
            width="stretch",
            height=500,
            column_order=[
                "url",
                "artwork_url",
                "name",
                "artists",
                "popularity",
                "release_date",
                "isrc",
                "Stats"
            ],
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
        st.metric("Average Track Popularity", f"{avg_popularity:.1f}")

        col1, col2 = st.columns(2)

        with col1:
            fig_popularity = px.histogram(
                df,
                x="popularity",
                title="Distribution of Track Popularity",
                nbins=10
            )
            if primary_color:
                fig_popularity.update_traces(marker_color=primary_color)
            fig_popularity.update_layout(
                xaxis_title="Popularity Score",
                yaxis_title="Number of Tracks"
            )
            st.plotly_chart(fig_popularity, width="stretch")

        with col2:
            if not artist_stats.empty:
                top_by_tracks = artist_stats.nlargest(10, 'track_count')
                fig_tracks = px.bar(
                    top_by_tracks,
                    x='artist',
                    y='track_count',
                    title="Top 10 Artists by Number of Tracks",
                    labels={'artist': 'Artist', 'track_count': 'Number of Tracks'}
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
