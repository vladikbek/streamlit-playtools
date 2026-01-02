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
from app.config import AVAILABLE_MARKETS, BATCH_SIZE, MAX_WORKERS, VIRAL_PLAYLISTS
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

def search_playlists(sp: spotipy.Spotify, keyword: str, market: str) -> List[Dict]:
    """Search for playlists matching the keyword with advanced filtering"""
    # Always search for exactly 50 playlists
    results = sp.search(q=f'{keyword}', type='playlist', limit=50, market=market)
    all_playlists = results['playlists']['items']
    
    # Step 1: Remove Spotify's official playlists if not included
    if not include_spotify:
        filtered_playlists = [
            playlist for playlist in all_playlists
            if playlist['owner']['display_name'].lower() != 'spotify' and 
               playlist['owner']['id'].lower() != 'spotify'
        ]
    else:
        filtered_playlists = all_playlists
    
    # Step 2: Remove playlists with too many tracks
    size_filtered_playlists = [
        playlist for playlist in filtered_playlists
        if playlist['tracks']['total'] <= max_playlist_tracks
    ]
    
    # Step 3: If ignore_networks is enabled, keep only one playlist per owner
    if ignore_networks:
        seen_owners = set()
        owner_filtered_playlists = []
        # Sort by follower count to keep the most popular playlist from each owner
        sorted_by_followers = sorted(
            size_filtered_playlists,
            key=lambda x: x.get('followers', {}).get('total', 0),
            reverse=True
        )
        for playlist in sorted_by_followers:
            owner_id = playlist['owner']['id']
            if owner_id not in seen_owners:
                seen_owners.add(owner_id)
                owner_filtered_playlists.append(playlist)
        size_filtered_playlists = owner_filtered_playlists
    
    # Step 4: Sort by follower count and take top final_playlists playlists
    sorted_playlists = sorted(
        size_filtered_playlists,
        key=lambda x: x.get('followers', {}).get('total', 0),
        reverse=True
    )[:final_playlists]
    
    return sorted_playlists

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

def process_viral_playlist(args: Tuple[spotipy.Spotify, str, str]) -> Tuple[str, str, List[Dict]]:
    """Helper function to process a single viral playlist in parallel"""
    sp, playlist_uri, country_code = args
    try:
        playlist_id = playlist_uri.split(':')[-1]
        
        def fetch_viral_tracks():
            return sp.playlist_tracks(
                playlist_id,
                fields='items(track(name,id,artists(name)))',
                market=selected_markets[0]
            )
        
        # Get tracks with retry mechanism
        results = retry_spotify_api(fetch_viral_tracks)
        return playlist_uri, country_code, results['items']
    except Exception as e:
        st.warning(f"Could not fetch tracks from viral playlist {playlist_uri}: {str(e)}")
        return playlist_uri, country_code, []

def get_viral_playlists_tracks(sp: spotipy.Spotify) -> Dict[str, Dict]:
    """Get tracks from viral playlists using parallel processing"""
    viral_data = defaultdict(lambda: {'count': 0, 'countries': set()})

    try:
        # Create arguments for parallel processing
        playlist_args = [(sp, playlist_uri, country_code) for playlist_uri, country_code in VIRAL_PLAYLISTS]
        total_playlists = len(playlist_args)

        # Process playlists in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total_playlists)) as executor:
            futures = [executor.submit(process_viral_playlist, args) for args in playlist_args]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    playlist_uri, country_code, items = future.result()
                    # Process tracks from this playlist
                    for item in items:
                        if item['track']:
                            track = item['track']
                            artists = ', '.join([artist['name'] for artist in track['artists']])
                            track_key = f"{track['name']}|||{artists}"
                            viral_data[track_key]['count'] += 1
                            viral_data[track_key]['countries'].add(country_code)
                except Exception as e:
                    st.warning(f"Error processing viral playlist results: {str(e)}")
                    continue

    except Exception as e:
        st.warning(f"Error processing viral playlists: {str(e)}")
        return {}

    # Convert sets to sorted lists for better display
    return {k: {'count': v['count'], 'countries': sorted(v['countries'])} for k, v in viral_data.items()}

def calculate_virality_scores(tracks_info: Dict[str, Dict], viral_data: Dict[str, Dict]) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, int]]:
    """Calculate virality scores for tracks that appear in main results"""
    # Extract viral counts for tracks that appear in our results
    found_viral_tracks = {}
    viral_countries = {}
    viral_counts = {}
    
    for track_key, info in tracks_info.items():
        # Create lookup key for viral data
        viral_key = f"{info['name']}|||{info['artist']}"
        if viral_key in viral_data:
            found_viral_tracks[track_key] = viral_data[viral_key]['count']
            viral_countries[track_key] = viral_data[viral_key]['countries']
            viral_counts[track_key] = viral_data[viral_key]['count']
    
    # Calculate normalized scores (0-100) based on found tracks only
    viral_scores = {}
    if found_viral_tracks:
        max_viral_count = max(found_viral_tracks.values())
        viral_scores = {k: (v / max_viral_count) * 100 for k, v in found_viral_tracks.items()}
    
    return viral_scores, viral_countries, viral_counts

def get_top_tracks_from_multiple_playlists(sp: spotipy.Spotify, playlists: List[Dict]) -> List[Dict]:
    """Get top tracks from multiple playlists using parallel processing"""
    # Get viral tracks data if enabled (raw counts, without normalization)
    viral_data = get_viral_playlists_tracks(sp) if use_virality else {}
    
    track_info = defaultdict(lambda: {
        'count': 0,  # Regular playlist appearances only
        'popularity': 0, 
        'name': '', 
        'artist': '', 
        'release_date': '',
        'id': '',
        'artwork_url': '',
        'virality': 0,
        'viral_count': 0,
        'viral_countries': [],
        'album_id': '',
        'playlists': set()  # Regular playlists only
    })
    
    # Process playlists in parallel
    playlist_args = [(sp, playlist) for playlist in playlists]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_playlist_tracks, args) for args in playlist_args]
        
        for future in concurrent.futures.as_completed(futures):
            playlist, tracks = future.result()
            
            for item in tracks:
                track = item.get('track')
                if not track or not track.get('id'):  # Skip if track or track ID is missing
                    continue
                    
                # Safely get artist names
                artists = []
                for artist in track.get('artists', []):
                    if artist and isinstance(artist, dict) and artist.get('name'):
                        artists.append(artist['name'])
                
                if not artists:  # Skip if no valid artists found
                    continue
                    
                artists_str = ', '.join(artists)
                # Include album ID in the key when unique_album_tracks is enabled
                if unique_album_tracks:
                    track_key = f"{track.get('name', '')}|||{artists_str}|||{track.get('album', {}).get('id', '')}"
                else:
                    track_key = f"{track.get('name', '')}|||{artists_str}"
                
                # Get the smallest artwork URL (to save bandwidth)
                artwork_url = ''
                if track.get('album', {}).get('images'):
                    artwork_url = track['album']['images'][-1]['url']
                
                # Update track information with all data we need
                current_popularity = track_info[track_key].get('popularity', 0)
                track_info[track_key].update({
                    'name': track.get('name', ''),
                    'artist': artists_str,
                    'id': track['id'],
                    'popularity': max(current_popularity, track.get('popularity', 0)),
                    'release_date': track.get('album', {}).get('release_date', ''),
                    'count': track_info[track_key]['count'] + 1,  # Regular playlist count
                    'artwork_url': artwork_url,
                    'album_id': track.get('album', {}).get('id', ''),
                    'playlists': track_info[track_key].get('playlists', set()) | {playlist['name']}
                })
                
                # Add viral data if enabled (kept separate from regular playlist data)
                if use_virality:
                    track_name_key = f"{track.get('name', '')}|||{artists_str}"
                    track_info[track_key]['viral_countries'] = viral_data.get(track_name_key, {}).get('countries', [])
                    track_info[track_key]['viral_count'] = viral_data.get(track_name_key, {}).get('count', 0)
                    track_info[track_key]['virality'] = viral_data.get(track_name_key, {}).get('count', 0)
    
    # Filter out tracks that appear in only one regular playlist
    if not unique_album_tracks:
        track_info = {k: v for k, v in track_info.items() if v['count'] > 1}
    
    if not track_info:
        return []
    
    # Calculate virality scores only for tracks in our results
    viral_scores = {}
    viral_countries = {}
    viral_counts = {}
    if use_virality:
        viral_scores, viral_countries, viral_counts = calculate_virality_scores(track_info, viral_data)
    
    # Calculate scores using only regular playlist appearances for appearance_score
    max_count = max(info['count'] for info in track_info.values())
    tracks_list = []
    
    for info in track_info.values():
        if all(key in info for key in ['name', 'artist', 'popularity', 'release_date']):
            popularity_score = info['popularity']
            freshness_score = calculate_freshness(info['release_date'])
            appearance_score = calculate_appearance_score(info['count'], max_count)  # Based on regular playlists only
            
            # Get virality score if enabled (now properly normalized)
            virality_score = 0
            if use_virality:
                # Always use name+artist for viral lookup regardless of unique_album_tracks setting
                track_key = f"{info['name']}|||{info['artist']}"
                virality_score = viral_scores.get(track_key, 0)
            
            # Apply threshold filters (using regular playlist count)
            if (popularity_score < popularity_range[0] or 
                popularity_score > popularity_range[1] or
                freshness_score < min_freshness or
                (not unique_album_tracks and info['count'] < min_playlists)):
                continue

            # Apply release date filter if specified
            if min_release_date:
                release_date = info['release_date']
                if len(release_date) == 4:
                    release_date = f"{release_date}-01-01"
                elif len(release_date) == 7:
                    release_date = f"{release_date}-01"
                try:
                    track_date = datetime.strptime(release_date, '%Y-%m-%d').date()
                    if track_date < min_release_date:
                        continue
                except ValueError:
                    continue

            # Apply virality filter if enabled
            if use_virality and (virality_score < virality_range[0] or virality_score > virality_range[1]):
                continue
            
            total_score = (
                popularity_score * weight_popularity +
                freshness_score * weight_freshness +
                appearance_score * weight_appearances +
                virality_score * weight_virality
            )
            
            track_data = {
                'artwork_url': info['artwork_url'],
                'name': info['name'],
                'artist': info['artist'],
                'popularity': info['popularity'],
                'freshness': freshness_score,
                'occurrence': info['count'],
                'release_date': info['release_date'],
                'total_score': total_score,
                'url': f"spotify:track:{info['id']}",
                'album_id': info['album_id'],
                'playlists': info['playlists']
            }
            
            # Only add virality data if enabled
            if use_virality:
                track_data['virality'] = virality_score
                track_data['viral_count'] = viral_counts.get(track_key, 0)
                track_data['viral_countries'] = viral_countries.get(track_key, [])
                
            tracks_list.append(track_data)
    
    # First, aggregate total occurrences across all versions of each song (using regular playlist count only)
    song_occurrences = defaultdict(int)
    for track in tracks_list:
        song_key = f"{track['name']}|||{track['artist']}"
        song_occurrences[song_key] += track['occurrence']
    
    # Filter out songs that don't meet the minimum playlist appearances threshold
    filtered_tracks = []
    for track in tracks_list:
        song_key = f"{track['name']}|||{track['artist']}"
        if song_occurrences[song_key] >= min_playlists:
            # Add viral-only filter
            if use_virality and viral_only and track.get('virality', 0) == 0:
                continue
            filtered_tracks.append(track)
            
    # Sort by total score
    filtered_tracks.sort(key=lambda x: x['total_score'], reverse=True)
    
    # If unique album tracks is enabled, keep only the highest scoring track per song
    if unique_album_tracks:
        seen_songs = {}
        for track in filtered_tracks:
            album_key = track['album_id']
            if album_key not in seen_songs or track['total_score'] > seen_songs[album_key]['total_score']:
                seen_songs[album_key] = track
        filtered_tracks = list(seen_songs.values())
    
    # Return top tracks
    return filtered_tracks[:top_tracks_limit]

def search_playlists_parallel(args: Tuple[spotipy.Spotify, str, str]) -> List[Dict]:
    """Search for playlists with keyword and market in parallel"""
    sp, keyword, market = args
    playlists = search_playlists(sp, keyword, market)
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

st.title(":material/playlist_add: Generate Playlist")
st.write("Find the most popular tracks across user-created playlists!")

# Initialize Spotify client
sp = setup_spotify()

# Sidebar for configuration
st.sidebar.title("Settings")

# Default preset values
DEFAULT_PRESET = {
    'selected_markets': ['US'],
    'ignore_networks': False,
    'include_spotify': False,
    'show_label_info': False,
    'unique_album_tracks': False,
    'use_virality': False,
    'weight_popularity': 0.4,
    'weight_freshness': 0.5,
    'weight_appearances': 0.2,
    'weight_virality': 0.1,
    'max_playlist_tracks': 500,
    'final_playlists': 25,
    'top_tracks_limit': 100,
    'min_release_date': None,
    'popularity_range': (0, 100),
    'min_freshness': 0,
    'min_playlists': 2,
    'freshness_days': 365
}

# Initialize session state for preset values if not exists
if 'preset_values' not in st.session_state:
    st.session_state.preset_values = DEFAULT_PRESET.copy()

# Presets
st.sidebar.subheader("Presets")

def update_preset_values():
    """Update session state values when preset changes"""
    preset = st.session_state.preset
    
    if preset == "Default":
        st.session_state.preset_values = DEFAULT_PRESET.copy()
    elif preset == "Fresh Hits":
        st.session_state.preset_values = {
            'selected_markets': ['US'],
            'ignore_networks': True,
            'include_spotify': False,
            'show_label_info': False,
            'unique_album_tracks': True,
            'use_virality': False,
            'weight_popularity': 0.6,
            'weight_freshness': 0.7,
            'weight_appearances': 0.2,
            'weight_virality': 0.1,
            'max_playlist_tracks': 500,
            'final_playlists': 50,
            'top_tracks_limit': 100,
            'min_release_date': None,
            'popularity_range': (40, 100),
            'min_freshness': 1,
            'min_playlists': 2,
            'freshness_days': 90
        }

preset = st.sidebar.selectbox(
    "Choose a preset",
    options=["Default", "Fresh Hits"],
    key="preset",
    on_change=update_preset_values,
    help="Select a preset configuration or customize your own settings"
)

st.sidebar.markdown("###")

# Basic settings
st.sidebar.subheader("Basic Settings")

selected_markets = st.sidebar.multiselect(
    "Markets",
    options=list(AVAILABLE_MARKETS.keys()),
    default=st.session_state.preset_values['selected_markets'],
    format_func=lambda x: f"{x} - {AVAILABLE_MARKETS[x]}",
    help="Select markets to search playlists in"
)

# Add checkbox for ignoring playlist networks
ignore_networks = st.sidebar.checkbox(
    "Ignore Playlist Networks",
    value=st.session_state.preset_values['ignore_networks'],
    help="Keep only one playlist per creator to prevent network bias"
)

# Add checkbox for including Spotify playlists
include_spotify = st.sidebar.checkbox(
    "Include Spotify's Playlists",
    value=st.session_state.preset_values['include_spotify'],
    help="Include official playlists created by Spotify"
)

st.sidebar.markdown("###")

# Add checkbox for label information
show_label_info = st.sidebar.checkbox(
    "Show Record Label",
    value=st.session_state.preset_values['show_label_info'],
    help="Display record label information for each track"
)

# Add checkbox for unique album tracks
unique_album_tracks = st.sidebar.checkbox(
    "Remove Versions",
    value=st.session_state.preset_values['unique_album_tracks'],
    help="Keep only one version of each song"
)

st.sidebar.markdown("###")

# Add checkbox for virality feature
use_virality = st.sidebar.checkbox(
    "Check Viral Charts",
    value=st.session_state.preset_values['use_virality'],
    help="Include virality score from Spotify's Top 50 Viral playlists in ranking"
)

st.sidebar.markdown("#")  

# Scoring weights
st.sidebar.subheader("Scoring Weights")
weight_popularity = st.sidebar.slider("Popularity Weight", 0.0, 1.0, st.session_state.preset_values['weight_popularity'], 0.05)
weight_freshness = st.sidebar.slider("Freshness Weight", 0.0, 1.0, st.session_state.preset_values['weight_freshness'], 0.05)
weight_appearances = st.sidebar.slider("Appearances Weight", 0.0, 1.0, st.session_state.preset_values['weight_appearances'], 0.05)
if use_virality:
    weight_virality = st.sidebar.slider("Virality Weight", 0.0, 1.0, st.session_state.preset_values['weight_virality'], 0.05)
else:
    weight_virality = 0.0

# Normalize weights to sum to 1
total_weight = weight_popularity + weight_freshness + weight_appearances + (weight_virality if use_virality else 0)
if total_weight > 0:
    weight_popularity = weight_popularity / total_weight
    weight_freshness = weight_freshness / total_weight
    weight_appearances = weight_appearances / total_weight
    if use_virality:
        weight_virality = weight_virality / total_weight

st.sidebar.markdown("#")  

# Playlist filtering settings
st.sidebar.subheader("Playlist Filtering")
max_playlist_tracks = st.sidebar.slider(
    "Max tracks per playlist",
    min_value=50,
    max_value=1000,
    value=st.session_state.preset_values['max_playlist_tracks'],
    step=50,
    help="Playlists with more tracks than this will be filtered out"
)

final_playlists = st.sidebar.slider(
    "Number of playlists to analyze",
    min_value=5,
    max_value=50,
    value=st.session_state.preset_values['final_playlists'],
    step=5,
    help="Number of top playlists to analyze after filtering"
)

top_tracks_limit = st.sidebar.slider(
    "Number of top tracks to show",
    min_value=10,
    max_value=100,
    value=st.session_state.preset_values['top_tracks_limit'],
    step=10,
    help="Number of top tracks to display in results"
)

st.sidebar.markdown("#")  

# Threshold settings
st.sidebar.subheader("Threshold Settings")

# Add release date threshold
min_release_date = st.sidebar.date_input(
    "Minimum Release Date",
    value=st.session_state.preset_values['min_release_date'],
    help="Filter out tracks released before this date (leave empty to include all)"
)

popularity_range = st.sidebar.slider(
    "Popularity Range",
    min_value=0,
    max_value=100,
    value=st.session_state.preset_values['popularity_range'],
    step=5,
    help="Filter tracks within this popularity range"
)

min_freshness = st.sidebar.slider(
    "Minimum Freshness",
    min_value=0,
    max_value=100,
    value=st.session_state.preset_values['min_freshness'],
    step=5,
    help="Filter out tracks below this freshness score"
)

min_playlists = st.sidebar.slider(
    "Minimum Playlist Appearances",
    min_value=1,
    max_value=20,
    value=st.session_state.preset_values['min_playlists'],
    step=1,
    help="Filter out tracks that appear in fewer playlists"
)

freshness_days = st.sidebar.slider(
    "Freshness Days",
    min_value=30,
    max_value=730,
    value=st.session_state.preset_values['freshness_days'],
    step=30,
    help="Number of days to consider a track fresh (affects freshness score)"
)

if use_virality:
    virality_range = st.sidebar.slider(
        "Virality Range",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=5,
        help="Filter tracks within this virality range"
    )
    
    viral_only = st.sidebar.checkbox(
        "Show Only Viral Tracks",
        value=False,
        help="Show only tracks that appear in viral playlists"
    )

# Create input field for keyword
search_col1, search_col2 = st.columns([4, 1])  # Ratio 4:1 for input:button

with search_col1:
    keyword = st.text_input("Enter keywords to search for playlists (separate by comma):", "", label_visibility="collapsed")

with search_col2:
    search_button = st.button("Search", type="primary", width="stretch")

if search_button and keyword:
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
            (sp, keyword, market)
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
        
        # Display search summary
        st.info(f"Found {len(unique_playlists)} unique playlists across {len(keywords)} keywords and {len(selected_markets)} markets.")
    
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
        st.error("No tracks found matching the threshold criteria. Try adjusting the threshold settings.")
        st.stop()
        
    # Get label information if checkbox is checked
    if show_label_info:
        with st.spinner("Fetching label information..."):
            # Extract track IDs from spotify URLs
            track_ids = [url.split(':')[-1] for url in df['url']]
            album_details = get_album_details_batch(sp, track_ids)
            
            # Add label information to DataFrame
            df['Label'] = df['url'].apply(lambda x: album_details.get(x.split(':')[-1], {}).get('label', 'Unknown'))
    
    # Convert playlists set to list for each track
    df['playlists'] = df['playlists'].apply(list)
    
    # Reorder columns and rename them for better display
    columns = ['artwork_url', 'name', 'artist']
    if show_label_info:
        columns.append('Label')
    columns.extend(['total_score', 'popularity', 'freshness', 'occurrence', 'playlists'])
    if use_virality:
        columns.extend(['viral_count', 'viral_countries'])
    columns.extend(['release_date', 'url'])
    
    df = df[columns]
    column_names = {
        'artwork_url': 'Artwork',
        'name': 'Track Name',
        'artist': 'Artists',
        'total_score': 'Total Score',
        'popularity': 'Popularity',
        'freshness': 'Freshness',
        'occurrence': 'Playlist #',
        'playlists': 'Found In',
        'viral_count': 'Viral #',
        'viral_countries': 'Viral In',
        'release_date': 'Release Date',
        'url': 'URL'
    }
    df.columns = [column_names.get(col, col) for col in df.columns]
    
    # Round numerical columns
    df['Total Score'] = df['Total Score'].round(1)
    df['Freshness'] = df['Freshness'].round(1)
    
    # Convert comma-separated artists string to list
    df['Artists'] = df['Artists'].str.split(', ')
    
    # Sort DataFrame by Total Score in descending order by default
    df = df.sort_values('Total Score', ascending=False)
    
    # Add Stats URL column
    df['Stats'] = df['URL'].apply(lambda x: f"https://www.mystreamcount.com/track/{x.split(':')[-1]}")
    
    # Display results
    st.subheader(f"Top {len(df)} Tracks")
    
    # Reset index to start from 1
    df.index = range(1, len(df) + 1)
    
    column_config = {
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
        "URL": st.column_config.LinkColumn(
            "Play",
            display_text="▶️ Play",
            help="Click to open in Spotify desktop/mobile app",
            width="small"
        ),
        "Stats": st.column_config.LinkColumn(
            "Stats",
            display_text="📊 Stats",
            help="Click to view track statistics",
            width="small"
        ),
        "Total Score": st.column_config.NumberColumn(
            "Total Score",
            help="Aggregated ranking score based on popularity, freshness, and playlist appearances. Click to sort.",
            format="%.1f"
        ),
        "Popularity": st.column_config.NumberColumn(
            "Popularity",
            help="Spotify's popularity score (0-100). Click to sort.",
            format="%d"
        ),
        "Freshness": st.column_config.NumberColumn(
            "Freshness",
            help="Score based on release date (0-100). Click to sort.",
            format="%.1f"
        ),
        "Playlist #": st.column_config.NumberColumn(
            "Playlist #",
            width="small",
            help="Number of regular playlists the track appears in. Click to sort.",
            format="%d"
        ),
        "Found In": st.column_config.ListColumn(
            "Found In",
            width="small",
            help="List of regular playlists where this track was found"
        ),
        "Artists": st.column_config.ListColumn(
            "Artists",
            width="medium",
            help="List of artists who performed this track"
        ),
        "Release Date": st.column_config.DateColumn(
            "Release Date",
            help="Track's release date. Click to sort."
        ),
        "Viral #": st.column_config.NumberColumn(
            "Viral #",
            help="Number of viral playlists the track appears in. Click to sort.",
            format="%d",
            width="small"
        ),
        "Viral In": st.column_config.ListColumn(
            "Viral In",
            help="Countries where the track appears in viral playlists",
            width="small"
        )
    }
    
    if show_label_info:
        column_config["Label"] = st.column_config.TextColumn(
            "Label",
            help="Record label that released the track. Click to sort.",
            width="medium"
        )
    
    st.dataframe(
        df,
        width="stretch",
        column_config=column_config,
        hide_index=False
    )

    # Create an expander for analytics
    with st.expander("View Analytics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Popularity vs. Freshness scatter plot
            fig_scatter = px.scatter(
                df,
                x="Freshness",
                y="Popularity",
                color= "Total Score",
                hover_data=["Track Name", "Artists"],
                title="Popularity vs. Freshness"
            )
            st.plotly_chart(fig_scatter, width="stretch")
        
        with col2:
            all_artists = [artist for artists_list in df['Artists'] for artist in artists_list]
            artist_counts = pd.Series(all_artists).value_counts().head(10)
            
            fig_artists = px.bar(
                x=artist_counts.index,
                y=artist_counts.values,
                title="Top 10 Artists",
                labels={'x': 'Artist', 'y': 'Number of Tracks'}
            )
            st.plotly_chart(fig_artists, width="stretch")

        # Add record label pie chart if label information is enabled
        if show_label_info:
            # Get top 5 labels and their counts
            label_counts = df['Label'].value_counts().head(5)
            
            # Create pie chart
            fig_labels = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="Top 5 Record Labels"
            )
            
            # Update layout for better readability
            fig_labels.update_traces(textposition='inside', textinfo='percent+label')
            fig_labels.update_layout(showlegend=True)
            
            # Display the chart in full width
            st.plotly_chart(fig_labels, width="stretch", theme="streamlit")

        # Add playlist metrics table
        st.subheader("Playlist Metrics")
        
        # Create a DataFrame with playlist metrics
        playlist_metrics = []
        for playlist in unique_playlists:
            # Get tracks that appear in this playlist
            playlist_tracks = df[df['Found In'].apply(lambda x: playlist['name'] in x)]
            
            metrics = {
                'Playlist Name': playlist['name'],
                'Creator': playlist['owner']['display_name'],
                'Market': playlist['source_market'],
                'Keyword': playlist['source_keyword'],
                'Tracks Found': len(playlist_tracks),
                'Total Score': playlist_tracks['Total Score'].sum(),
                'Avg Score': round(playlist_tracks['Total Score'].mean(), 2) if len(playlist_tracks) > 0 else 0,
                # 'Followers': playlist['followers']['total']
            }
            playlist_metrics.append(metrics)
            
        playlist_df = pd.DataFrame(playlist_metrics)
        
        # Configure columns for the playlist metrics table
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
            # "Followers": st.column_config.NumberColumn(
            #     "Followers",
            #     help="Number of playlist followers",
            #     format="%d"
            # )
        }
        
        # Sort by Total Score by default
        playlist_df = playlist_df.sort_values('Total Score', ascending=False)
        
        # Display the playlist metrics table
        st.dataframe(
            playlist_df,
            width="stretch",
            column_config=playlist_column_config,
            hide_index=True
        )

    # Create an expander for track URIs
    with st.expander("Track URIs", expanded=False):
        # Extract all URIs and join them with newlines
        track_uris = '\n'.join(df['URL'].tolist())
        st.code(track_uris, language='markdown')
