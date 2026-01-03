import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import List, Dict, Tuple
from collections import defaultdict
import concurrent.futures
from datetime import datetime, date
import pandas as pd
import streamlit as st
import plotly.express as px
from app.config import AVAILABLE_MARKETS, BATCH_SIZE, MAX_WORKERS
import time

# Configuration variables
MAX_WORKERS = 10           # Maximum number of parallel workers
BATCH_SIZE = 20           # Spotify API allows up to 20 items in batch requests

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

def calculate_freshness(release_date: str) -> float:
    """Calculate freshness score (0-100) based on release date"""
    try:
        # Handle different date formats from Spotify API
        if len(release_date) == 4:  # Year only
            release_date = f"{release_date}-01-01"
        elif len(release_date) == 7:  # Year-Month
            release_date = f"{release_date}-01"
        
        release_date = datetime.strptime(release_date, '%Y-%m-%d').date()
        today = date.today()
        days_since_release = (today - release_date).days
        
        # Score from 0 to 100, with 0 for songs older than freshness_days
        if days_since_release >= freshness_days:
            return 0
        return max(0, 100 * (1 - days_since_release / freshness_days))
    except (ValueError, TypeError):
        return 0  # Return 0 if date parsing fails

def calculate_appearance_score(count: int, max_count: int) -> float:
    """Calculate normalized appearance score (0-100) based on playlist count"""
    if max_count == 0:
        return 0
    return (count / max_count) * 100

def search_playlists(sp: spotipy.Spotify, keyword: str, market: str, limit: int) -> List[Dict]:
    """Search for playlists matching the keyword."""
    results = sp.search(q=f'{keyword}', type='playlist', limit=limit, market=market)
    return results['playlists']['items']

def is_spotify_playlist(playlist: Dict) -> bool:
    owner = playlist.get('owner') or {}
    owner_id = (owner.get('id') or '').lower()
    owner_name = (owner.get('display_name') or '').lower()
    return owner_id == 'spotify' or owner_name == 'spotify'

def apply_playlist_filters(
    playlists: List[Dict],
    remove_spotify: bool,
    remove_networks: bool,
    max_playlist_tracks: int,
    final_playlists: int
) -> List[Dict]:
    grouped_playlists = defaultdict(list)
    for playlist in playlists:
        group_key = (playlist.get('source_keyword'), playlist.get('source_market'))
        grouped_playlists[group_key].append(playlist)

    filtered_playlists = []
    for group in grouped_playlists.values():
        group_filtered = group

        if remove_spotify:
            group_filtered = [playlist for playlist in group_filtered if not is_spotify_playlist(playlist)]

        group_filtered = [
            playlist for playlist in group_filtered
            if (playlist.get('tracks') or {}).get('total', 0) <= max_playlist_tracks
        ]

        if remove_networks:
            seen_owners = set()
            owner_filtered = []
            sorted_by_followers = sorted(
                group_filtered,
                key=lambda x: x.get('followers', {}).get('total', 0),
                reverse=True
            )
            for playlist in sorted_by_followers:
                owner = playlist.get('owner') or {}
                owner_id = owner.get('id') or owner.get('display_name') or ''
                if owner_id not in seen_owners:
                    seen_owners.add(owner_id)
                    owner_filtered.append(playlist)
            group_filtered = owner_filtered

        sorted_playlists = sorted(
            group_filtered,
            key=lambda x: x.get('followers', {}).get('total', 0),
            reverse=True
        )
        filtered_playlists.extend(sorted_playlists[:final_playlists])

    seen_ids = set()
    unique_filtered = []
    for playlist in filtered_playlists:
        playlist_id = playlist.get('id')
        if not playlist_id or playlist_id in seen_ids:
            continue
        seen_ids.add(playlist_id)
        unique_filtered.append(playlist)

    return unique_filtered

def retry_spotify_api(func, max_retries=3, initial_delay=1):
    """Retry Spotify API calls with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "Read timed out" in str(e) and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(delay)
                continue
            raise e

def get_tracks_batch(sp: spotipy.Spotify, playlist: Dict) -> Tuple[Dict, List[Dict]]:
    """Get all tracks from a playlist efficiently using fields filter and pagination"""
    fields = 'items(track(name,id,artists(name),album(id,release_date,images),popularity)),next'
    tracks = []
    
    def fetch_tracks():
        return sp.playlist_tracks(
            playlist['id'],
            fields=fields,
            market=selected_markets[0]
        )
    
    # Get initial results with retry
    results = retry_spotify_api(fetch_tracks)
    
    # Get all tracks using pagination
    tracks.extend(results['items'])
    while results['next']:
        def fetch_next():
            return sp.next(results)
        results = retry_spotify_api(fetch_next)
        tracks.extend(results['items'])
    
    return playlist, tracks

def process_playlist_tracks(args: Tuple[spotipy.Spotify, Dict]) -> Tuple[Dict, List[Dict]]:
    """Helper function to process playlist tracks in parallel"""
    sp, playlist = args
    return get_tracks_batch(sp, playlist)

def get_top_tracks_from_multiple_playlists(sp: spotipy.Spotify, playlists: List[Dict]) -> List[Dict]:
    """Get aggregated track data from multiple playlists using parallel processing."""
    track_info = defaultdict(lambda: {
        'popularity': 0,
        'name': '',
        'artist': '',
        'release_date': '',
        'id': '',
        'artwork_url': '',
        'album_id': '',
        'playlist_ids': set()
    })

    playlist_args = [(sp, playlist) for playlist in playlists]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_playlist_tracks, args) for args in playlist_args]

        for future in concurrent.futures.as_completed(futures):
            playlist, tracks = future.result()
            playlist_id = playlist.get('id')
            if not playlist_id:
                continue

            for item in tracks:
                track = item.get('track')
                if not track or not track.get('id'):
                    continue

                artists = []
                for artist in track.get('artists', []):
                    if artist and isinstance(artist, dict) and artist.get('name'):
                        artists.append(artist['name'])

                if not artists:
                    continue

                artists_str = ', '.join(artists)
                album_info = track.get('album', {}) or {}
                album_id = album_info.get('id', '')
                track_name = track.get('name', '')
                track_key = f"{track_name}|||{artists_str}|||{album_id}"

                images = album_info.get('images') or []
                artwork_url = images[-1]['url'] if images else ''

                track_popularity = track.get('popularity', 0)
                current_popularity = track_info[track_key].get('popularity', 0)

                if not track_info[track_key]['id'] or track_popularity >= current_popularity:
                    track_info[track_key].update({
                        'id': track['id'],
                        'popularity': track_popularity,
                        'release_date': album_info.get('release_date', ''),
                        'artwork_url': artwork_url
                    })

                if not track_info[track_key].get('artwork_url') and artwork_url:
                    track_info[track_key]['artwork_url'] = artwork_url

                track_info[track_key]['name'] = track_name
                track_info[track_key]['artist'] = artists_str
                track_info[track_key]['album_id'] = album_id
                track_info[track_key]['playlist_ids'].add(playlist_id)

    if not track_info:
        return []

    tracks_list = []
    for track_key, info in track_info.items():
        if all(key in info for key in ['name', 'artist', 'popularity', 'release_date', 'id']):
            tracks_list.append({
                'track_key': track_key,
                'song_key': f"{info['name']}|||{info['artist']}",
                'artwork_url': info['artwork_url'],
                'name': info['name'],
                'artist': info['artist'],
                'popularity': info['popularity'],
                'release_date': info['release_date'],
                'url': f"spotify:track:{info['id']}",
                'album_id': info['album_id'],
                'playlist_ids': info['playlist_ids']
            })

    return tracks_list

def search_playlists_parallel(args: Tuple[spotipy.Spotify, str, str, int]) -> List[Dict]:
    """Search for playlists with keyword and market in parallel"""
    sp, keyword, market, limit = args
    playlists = search_playlists(sp, keyword, market, limit)
    # Add source information
    for playlist in playlists:
        playlist['source_keyword'] = keyword
        playlist['source_market'] = market
    return playlists

def process_album_details(args: Tuple[spotipy.Spotify, List[str]]) -> Dict[str, Dict]:
    """Helper function to process album details in parallel using Get Several Albums endpoint"""
    sp, track_batch = args
    album_details = {}
    
    # First get the tracks to extract album IDs
    tracks = sp.tracks(track_batch)['tracks']
    
    # Collect unique album IDs
    album_ids = []
    track_to_album = {}  # Map track IDs to their album IDs
    for track in tracks:
        if track and track['album']:
            album_id = track['album']['id']
            album_ids.append(album_id)
            track_to_album[track['id']] = album_id
    
    # Get unique album IDs (a track might be from the same album as another track)
    unique_album_ids = list(set(album_ids))
    
    # Fetch albums in batches of 20 (Spotify API limit)
    for i in range(0, len(unique_album_ids), 20):
        album_batch = unique_album_ids[i:i + 20]
        albums_response = sp.albums(album_batch)
        
        # Create mapping of album ID to label
        album_to_label = {
            album['id']: album.get('label', 'Unknown')
            for album in albums_response['albums']
            if album is not None
        }
        
        # Map track IDs to their album labels
        for track_id, album_id in track_to_album.items():
            album_details[track_id] = {
                'label': album_to_label.get(album_id, 'Unknown')
            }
    
    return album_details

def get_album_details_batch(sp: spotipy.Spotify, track_ids: List[str]) -> Dict[str, Dict]:
    """Get album details for a batch of tracks using parallel processing"""
    album_details = {}
    
    # Split track_ids into batches
    batches = [track_ids[i:i + BATCH_SIZE] for i in range(0, len(track_ids), BATCH_SIZE)]
    total_batches = len(batches)
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total_batches)) as executor:
        batch_args = [(sp, batch) for batch in batches]
        futures = [executor.submit(process_album_details, args) for args in batch_args]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_details = future.result()
                album_details.update(batch_details)
            except Exception as e:
                st.warning(f"Failed to fetch some album details: {str(e)}")
                continue
    
    return album_details

def select_top_album_versions(track_df: pd.DataFrame) -> pd.DataFrame:
    grouped_df = track_df.copy()
    grouped_df['version_group'] = grouped_df['album_id'].fillna('')
    grouped_df['version_group'] = grouped_df['version_group'].where(
        grouped_df['version_group'].astype(str).str.len() > 0,
        grouped_df['song_key']
    )

    grouped_df = grouped_df.sort_values(
        ['total_score', 'popularity', 'name', 'url'],
        ascending=[False, False, True, True],
        kind='mergesort'
    )
    grouped_df = grouped_df.drop_duplicates(subset=['version_group'], keep='first')
    return grouped_df.drop(columns=['version_group'])

st.title(":material/playlist_add: Generate Playlist", anchor=False)
st.caption("Find the most popular tracks across user-created playlists.")

# Initialize Spotify client
sp = setup_spotify()

# Sidebar for configuration
st.sidebar.title("Options")

with st.sidebar.container(border=True):
    st.subheader("Basic Settings")

    remove_networks = st.checkbox(
        "Remove Playlist Networks",
        value=False,
        help="Keep only one playlist per creator to prevent network bias"
    )

    remove_spotify = st.checkbox(
        "Remove Spotify's Playlists",
        value=False,
        help="Hide official playlists created by Spotify"
    )

    unique_album_tracks = st.checkbox(
        "Remove Song Versions",
        value=False,
        help="Keep only one version of each song"
    )

    st.markdown("#")

    st.subheader("Rating")
    weight_popularity = st.slider("Popularity", 0.0, 1.0, 0.4, 0.05)
    weight_freshness = st.slider("Freshness", 0.0, 1.0, 0.5, 0.05)
    weight_appearances = st.slider("Appearances", 0.0, 1.0, 0.2, 0.05)

    total_weight = weight_popularity + weight_freshness + weight_appearances
    if total_weight > 0:
        weight_popularity = weight_popularity / total_weight
        weight_freshness = weight_freshness / total_weight
        weight_appearances = weight_appearances / total_weight

    st.markdown("#")

    st.subheader("Filters")
    min_playlists = st.slider(
        "Minimum Playlists",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
        help="Filter out tracks that appear in fewer playlists"
    )

    popularity_range = st.slider(
        "Popularity Range",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=5,
        help="Filter tracks within this popularity range"
    )

    min_release_date = st.date_input(
        "Minimum Release Date",
        value=None,
        help="Filter out tracks released before this date (leave empty to include all)"
    )

    freshness_days = st.slider(
        "Freshness Days",
        min_value=30,
        max_value=730,
        value=365,
        step=30,
        help="Number of days to consider a track fresh (affects freshness score)"
    )

    min_freshness = st.slider(
        "Minimum Freshness",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Filter out tracks below this freshness score"
    )

    st.markdown("#")

    st.subheader("Other")
    max_playlist_tracks = st.slider(
        "Max tracks per playlist",
        min_value=50,
        max_value=1000,
        value=500,
        step=50,
        help="Playlists with more tracks than this will be filtered out"
    )

    search_limit_max = int(st.session_state.get("generate_search_limit", 50))
    search_limit_max = max(1, min(50, search_limit_max))
    last_search_limit = st.session_state.get("generate_search_limit_max")
    if "generate_final_playlists" not in st.session_state:
        st.session_state.generate_final_playlists = search_limit_max
    else:
        if last_search_limit is not None and st.session_state.generate_final_playlists == last_search_limit:
            st.session_state.generate_final_playlists = search_limit_max
        st.session_state.generate_final_playlists = max(
            1,
            min(search_limit_max, st.session_state.generate_final_playlists)
        )
    st.session_state.generate_search_limit_max = search_limit_max

    final_playlists = st.slider(
        "Number of playlists to analyze",
        min_value=1,
        max_value=search_limit_max,
        value=st.session_state.generate_final_playlists,
        step=1,
        key="generate_final_playlists",
        help="Number of top playlists to analyze per keyword and market after filtering"
    )

    top_tracks_placeholder = st.empty()
# Create input field for keyword
with st.container(border=True):
    with st.form("generate_playlist_form", border=False):
        search_col1, search_col2, search_col3 = st.columns([8, 2, 3])

        with search_col1:
            keyword = st.text_input(
                "Enter keywords to search for playlists (separate by comma):",
                "",
                label_visibility="collapsed",
                placeholder="Paste keywords, split with commas (e.g. phonk, hard techno)",
                icon=":material/link:"
            )

        with search_col2:
            search_limit = st.number_input(
                "Playlist limit",
                min_value=1,
                max_value=50,
                value=50,
                step=1,
                help="Maximum number of playlists to fetch per keyword and market",
                label_visibility="collapsed",
                icon=":material/post:",
                key="generate_search_limit"
            )

        with search_col3:
            search_button = st.form_submit_button(
                "Search",
                type="primary",
                icon=":material/search:",
                width="stretch"
            )

        with st.container(horizontal=True, vertical_alignment="center"):
            selected_markets = st.multiselect(
                "Markets",
                options=list(AVAILABLE_MARKETS.keys()),
                default=['US'],
                format_func=lambda x: f"{x} - {AVAILABLE_MARKETS[x]}",
                help="Select markets to search playlists in",
                label_visibility="collapsed",
                placeholder="Select markets to search playlists in"
            )

            show_label_info = st.checkbox(
                "Get Record Label",
                value=False,
                help="Display record label information for each track"
            )

if not selected_markets:
    selected_markets = ['US']

if search_button:
    if not keyword:
        st.error("Please enter at least one keyword.")
        st.stop()

    # Split keywords and clean them
    keywords = [k.strip() for k in keyword.split(',') if k.strip()]

    if not keywords:
        st.error("Please enter at least one keyword.")
        st.stop()

    all_playlists = []
    total_searches = len(keywords) * len(selected_markets)

    with st.spinner(f"Searching for playlists with {len(keywords)} keywords across {len(selected_markets)} markets..."):
        # Create all combinations of keyword and market
        search_args = [
            (sp, keyword, market, int(search_limit))
            for keyword in keywords
            for market in selected_markets
        ]

        # Process searches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total_searches)) as executor:
            futures = [executor.submit(search_playlists_parallel, args) for args in search_args]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    playlists = future.result()
                    all_playlists.extend(playlists)
                except Exception as e:
                    st.warning(f"A search failed: {str(e)}")
                    continue

        # Remove duplicates based on playlist ID
        seen_ids = set()
        unique_playlists = []
        for playlist in all_playlists:
            if playlist['id'] not in seen_ids:
                seen_ids.add(playlist['id'])
                unique_playlists.append(playlist)

    if not unique_playlists:
        st.error("No playlists found after applying filters. Try adjusting the filter settings or using different keywords.")
        st.stop()

    with st.spinner("Analyzing tracks across playlists..."):
        top_tracks = get_top_tracks_from_multiple_playlists(sp, unique_playlists)

    if not top_tracks:
        st.error("No common tracks found across playlists.")
        st.stop()

    # Convert to pandas DataFrame
    df = pd.DataFrame(top_tracks)

    if df.empty:
        st.error("No tracks found matching the search criteria.")
        st.stop()

    # Get label information if checkbox is checked
    if show_label_info:
        with st.spinner("Fetching label information..."):
            # Extract track IDs from spotify URLs
            track_ids = [url.split(':')[-1] for url in df['url']]
            album_details = get_album_details_batch(sp, track_ids)

            # Add label information to DataFrame
            df['label'] = df['url'].apply(lambda x: album_details.get(x.split(':')[-1], {}).get('label', 'Unknown'))

    st.session_state["generate_results"] = {
        "df": df,
        "keywords": keywords,
        "markets": selected_markets,
        "playlists": all_playlists
    }

generate_results = st.session_state.get("generate_results")
if generate_results:
    base_df = generate_results["df"].copy()
    keywords = generate_results["keywords"]
    result_markets = generate_results["markets"]
    all_playlists = generate_results["playlists"]

    if show_label_info and "label" not in base_df.columns:
        with st.spinner("Fetching label information..."):
            track_ids = [url.split(':')[-1] for url in base_df['url']]
            album_details = get_album_details_batch(sp, track_ids)
            base_df['label'] = base_df['url'].apply(
                lambda x: album_details.get(x.split(':')[-1], {}).get('label', 'Unknown')
            )
        st.session_state["generate_results"]["df"] = base_df

    active_playlists = apply_playlist_filters(
        all_playlists,
        remove_spotify,
        remove_networks,
        max_playlist_tracks,
        final_playlists
    )

    if not active_playlists:
        st.warning("No playlists left after applying playlist filters.")
        st.stop()

    playlist_lookup = {playlist['id']: playlist for playlist in active_playlists}
    active_playlist_ids = set(playlist_lookup.keys())

    track_df = base_df.copy()
    if 'song_key' not in track_df.columns:
        track_df['song_key'] = track_df['name'] + "|||" + track_df['artist']

    track_df['playlist_ids'] = track_df['playlist_ids'].apply(
        lambda value: set(value) if isinstance(value, (list, tuple)) else value
    )
    track_df['playlist_ids_filtered'] = track_df['playlist_ids'].apply(lambda ids: ids & active_playlist_ids)
    track_df['occurrence'] = track_df['playlist_ids_filtered'].apply(len)
    track_df = track_df[track_df['occurrence'] > 0]

    if track_df.empty:
        st.warning("No tracks found after applying playlist filters.")
        st.stop()

    if unique_album_tracks:
        track_df['song_occurrence'] = track_df.groupby('song_key')['occurrence'].transform('sum')
    else:
        track_df['song_occurrence'] = track_df['occurrence']

    track_df['freshness'] = track_df['release_date'].apply(calculate_freshness)

    filtered_df = track_df.copy()
    filtered_df = filtered_df[
        (filtered_df['popularity'] >= popularity_range[0]) &
        (filtered_df['popularity'] <= popularity_range[1])
    ]

    if min_release_date:
        filtered_df['release_date_parsed'] = pd.to_datetime(filtered_df['release_date'], errors='coerce')
        filtered_df = filtered_df[
            filtered_df['release_date_parsed'].dt.date >= min_release_date
        ]

    filtered_df = filtered_df[filtered_df['freshness'] >= min_freshness]
    filtered_df = filtered_df[filtered_df['song_occurrence'] >= min_playlists]

    if filtered_df.empty:
        st.warning("No tracks found after applying the filters.")
        st.stop()

    max_count = filtered_df['occurrence'].max() if len(filtered_df) > 0 else 0
    filtered_df['appearance_score'] = filtered_df['occurrence'].apply(
        lambda count: calculate_appearance_score(count, max_count)
    )

    filtered_df['total_score'] = (
        filtered_df['popularity'] * weight_popularity +
        filtered_df['freshness'] * weight_freshness +
        filtered_df['appearance_score'] * weight_appearances
    )

    filtered_df = filtered_df.sort_values('total_score', ascending=False)

    if unique_album_tracks:
        filtered_df = select_top_album_versions(filtered_df)

    if filtered_df.empty:
        st.warning("No tracks found after applying the filters.")
        st.stop()

    total_tracks = len(filtered_df)
    max_tracks = max(1, total_tracks)
    min_tracks = 1
    step_tracks = 10 if max_tracks >= 10 else 1
    default_tracks = 100 if max_tracks >= 100 else max_tracks
    current_tracks = st.session_state.get("generate_top_tracks_limit", default_tracks)
    current_tracks = max(min_tracks, min(max_tracks, current_tracks))
    st.session_state.generate_top_tracks_limit = current_tracks

    if max_tracks > 1:
        top_tracks_limit = top_tracks_placeholder.slider(
            "Number of top tracks to show",
            min_value=min_tracks,
            max_value=max_tracks,
            value=current_tracks,
            step=step_tracks,
            key="generate_top_tracks_limit",
            help="Number of top tracks to display in results"
        )
    else:
        top_tracks_limit = top_tracks_placeholder.number_input(
            "Number of top tracks to show",
            min_value=1,
            max_value=1,
            value=1,
            step=1,
            key="generate_top_tracks_limit_single",
            help="Number of top tracks to display in results"
        )
        st.session_state.generate_top_tracks_limit = 1

    filtered_df = filtered_df.head(top_tracks_limit)
    filtered_df['playlists'] = filtered_df['playlist_ids_filtered'].apply(
        lambda ids: [playlist_lookup[playlist_id]['name'] for playlist_id in ids if playlist_id in playlist_lookup]
    )

    # Reorder columns and rename them for better display
    columns = ['artwork_url', 'name', 'artist', 'release_date']
    if show_label_info:
        columns.insert(3, 'label')
    columns.extend(['total_score', 'popularity', 'freshness', 'occurrence', 'playlists', 'url'])

    display_df = filtered_df[columns].copy()
    column_names = {
        'artwork_url': 'Artwork',
        'name': 'Track Name',
        'artist': 'Artists',
        'label': 'Label',
        'total_score': 'Total Score',
        'popularity': 'Popularity',
        'freshness': 'Freshness',
        'occurrence': 'Playlist #',
        'playlists': 'Playlists',
        'release_date': 'Release Date',
        'url': 'URL'
    }
    display_df.columns = [column_names.get(col, col) for col in display_df.columns]

    display_df['Total Score'] = display_df['Total Score'].round(1)
    display_df['Freshness'] = display_df['Freshness'].round(1)
    display_df['Artists'] = display_df['Artists'].str.split(', ')
    display_df['Stats'] = display_df['URL'].apply(
        lambda x: f"https://www.mystreamcount.com/track/{x.split(':')[-1]}"
    )

    st.subheader(f"Found {len(display_df)} tracks", anchor=False)
    markets_label = ", ".join(AVAILABLE_MARKETS[m] for m in result_markets)
    keywords_label = ", ".join(keywords)
    st.caption(f"{keywords_label} in {markets_label}")

    primary_color = st.get_option("theme.primaryColor")
    tracks_tab, analytics_tab, playlists_tab, uris_tab = st.tabs(
        ["Tracks", "Analytics", "Playlists", "Track URIs"]
    )

    display_df.index = range(1, len(display_df) + 1)

    column_config = {
        "URL": st.column_config.LinkColumn(
            "Link",
            display_text=":material/open_in_new:",
            help="Click to open in Spotify desktop/mobile app",
            width="small"
        ),
        "Artwork": st.column_config.ImageColumn(
            "Artwork",
            width="small",
            help="Track's album artwork"
        ),
        "Track Name": st.column_config.TextColumn(
            "Track Name",
            help="Name of the track",
            width="medium"
        ),
        "Artists": st.column_config.ListColumn(
            "Artists",
            width="medium",
            help="List of artists who performed this track"
        ),
        "Total Score": st.column_config.NumberColumn(
            "Total Score",
            help="Aggregated ranking score based on popularity, freshness, and playlist appearances. Click to sort.",
            format="%.1f",
            width="small"
        ),
        "Popularity": st.column_config.NumberColumn(
            "Popularity",
            help="Spotify's popularity score (0-100). Click to sort.",
            format="%d",
            width="small"
        ),
        "Freshness": st.column_config.NumberColumn(
            "Freshness",
            help="Score based on release date (0-100). Click to sort.",
            format="%.1f",
            width="small"
        ),
        "Playlist #": st.column_config.NumberColumn(
            "Playlist #",
            help="Number of regular playlists the track appears in. Click to sort.",
            format="%d",
            width="small"
        ),
        "Playlists": st.column_config.ListColumn(
            "Playlists",
            help="List of regular playlists where this track was found",
            width="large"
        ),
        "Release Date": st.column_config.DateColumn(
            "Release Date",
            help="Track's release date. Click to sort."
        ),
        "Stats": st.column_config.LinkColumn(
            "Stats",
            display_text=":material/query_stats:",
            help="Click to view track statistics",
            width="small"
        )
    }

    if show_label_info:
        column_config["Label"] = st.column_config.TextColumn(
            "Label",
            help="Record label that released the track. Click to sort.",
            width="medium"
        )

    column_order = ["URL", "Artwork", "Track Name", "Artists"]
    if show_label_info:
        column_order.append("Label")
    column_order.append("Release Date")
    column_order.extend(["Total Score", "Popularity", "Freshness", "Playlist #", "Playlists", "Stats"])

    with tracks_tab:
        st.dataframe(
            display_df,
            width="stretch",
            height=500,
            column_order=column_order,
            column_config=column_config,
            hide_index=False
        )

    with analytics_tab:
        col1, col2 = st.columns(2)

        with col1:
            fig_scatter = px.scatter(
                display_df,
                x="Freshness",
                y="Popularity",
                color="Total Score",
                hover_data=["Track Name", "Artists"],
                title="Popularity vs. Freshness"
            )
            st.plotly_chart(fig_scatter, width="stretch")

        with col2:
            all_artists = [artist for artists_list in display_df['Artists'] for artist in artists_list]
            artist_counts = pd.Series(all_artists).value_counts().head(10)

            fig_artists = px.bar(
                x=artist_counts.index,
                y=artist_counts.values,
                title="Top 10 Artists",
                labels={'x': 'Artist', 'y': 'Number of Tracks'}
            )
            if primary_color:
                fig_artists.update_traces(marker_color=primary_color)
            st.plotly_chart(fig_artists, width="stretch")

        if show_label_info:
            label_counts = display_df['Label'].value_counts().head(5)
            fig_labels = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="Top 5 Record Labels"
            )
            fig_labels.update_traces(textposition='inside', textinfo='percent+label')
            fig_labels.update_layout(showlegend=True)
            st.plotly_chart(fig_labels, width="stretch", theme="streamlit")

    with playlists_tab:
        playlist_metrics = []
        for playlist in active_playlists:
            playlist_tracks = filtered_df[
                filtered_df['playlist_ids_filtered'].apply(lambda ids: playlist['id'] in ids)
            ]

            metrics = {
                'Playlist Name': playlist['name'],
                'Creator': playlist['owner']['display_name'],
                'Market': playlist['source_market'],
                'Keyword': playlist['source_keyword'],
                'Tracks Found': len(playlist_tracks),
                'Total Score': playlist_tracks['total_score'].sum(),
                'Avg Score': round(playlist_tracks['total_score'].mean(), 2) if len(playlist_tracks) > 0 else 0,
            }
            playlist_metrics.append(metrics)

        playlist_df = pd.DataFrame(playlist_metrics)
        playlist_column_config = {
            "Playlist Name": st.column_config.TextColumn(
                "Playlist Name",
                help="Name of the playlist",
                width="medium"
            ),
            "Creator": st.column_config.TextColumn(
                "Creator",
                help="Playlist creator",
                width="medium"
            ),
            "Market": st.column_config.TextColumn(
                "Market",
                help="Market where the playlist was found",
                width="small"
            ),
            "Keyword": st.column_config.TextColumn(
                "Keyword",
                help="Search keyword that found this playlist",
                width="small"
            ),
            "Tracks Found": st.column_config.NumberColumn(
                "Tracks Found",
                help="Number of tracks from this playlist that made it to the results",
                width="small"
            ),
            "Total Score": st.column_config.NumberColumn(
                "Total Score",
                help="Sum of scores for all tracks from this playlist",
                format="%.1f"
            ),
            "Avg Score": st.column_config.NumberColumn(
                "Avg Score",
                help="Average score of tracks from this playlist",
                format="%.2f"
            ),
        }

        playlist_df = playlist_df.sort_values('Total Score', ascending=False)
        st.dataframe(
            playlist_df,
            width="stretch",
            column_config=playlist_column_config,
            hide_index=True
        )

    with uris_tab:
        track_uris = '\n'.join(display_df['URL'].tolist())
        st.code(track_uris, language='markdown')
else:
    max_tracks = 100
    min_tracks = 1
    step_tracks = 10
    default_tracks = 100
    current_tracks = st.session_state.get("generate_top_tracks_limit", default_tracks)
    current_tracks = max(min_tracks, min(max_tracks, current_tracks))
    st.session_state.generate_top_tracks_limit = current_tracks

    top_tracks_placeholder.slider(
        "Number of top tracks to show",
        min_value=min_tracks,
        max_value=max_tracks,
        value=current_tracks,
        step=step_tracks,
        key="generate_top_tracks_limit",
        help="Number of top tracks to display in results"
    )
