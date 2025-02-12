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
from config import AVAILABLE_MARKETS, BATCH_SIZE, MAX_WORKERS

# Global variable declarations
global MAX_PLAYLIST_TRACKS, FINAL_PLAYLISTS_LIMIT, TOP_TRACKS_LIMIT
global WEIGHT_POPULARITY, WEIGHT_FRESHNESS, WEIGHT_APPEARANCES, WEIGHT_VIRALITY
global MARKET, FRESHNESS_DAYS

# Configuration variables
MAX_PLAYLIST_TRACKS = 300   # Maximum number of tracks in a playlist to consider
FINAL_PLAYLISTS_LIMIT = 25  # Number of top playlists to analyze after filtering
MARKET = 'US'              # Market for search
TOP_TRACKS_LIMIT = 100     # Default number of top tracks to return
MAX_WORKERS = 10           # Maximum number of parallel workers
BATCH_SIZE = 20           # Spotify API allows up to 20 items in batch requests
FRESHNESS_DAYS = 365      # Number of days to consider a track "fresh" (default: 1 year)

# Weighting configuration
WEIGHT_POPULARITY = 0.4    # Weight for popularity score (0-100)
WEIGHT_FRESHNESS = 0.5    # Weight for freshness score (0-100)
WEIGHT_APPEARANCES = 0.2   # Weight for playlist appearances
WEIGHT_VIRALITY = 0.3     # Weight for virality score (0-100)

def setup_spotify():
    """Initialize Spotify client with credentials from .env file"""
    load_dotenv()
    auth_manager = SpotifyClientCredentials()
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
        
        # Score from 0 to 100, with 0 for songs older than FRESHNESS_DAYS
        if days_since_release >= FRESHNESS_DAYS:
            return 0
        return max(0, 100 * (1 - days_since_release / FRESHNESS_DAYS))
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
    results = sp.search(q=f'"{keyword}"', type='playlist', limit=50, market=market)
    all_playlists = results['playlists']['items']
    
    # Step 1: Remove Spotify's official playlists
    filtered_playlists = [
        playlist for playlist in all_playlists
        if playlist['owner']['display_name'].lower() != 'spotify' and 
           playlist['owner']['id'].lower() != 'spotify'
    ]
    
    # Step 2: Remove playlists with too many tracks
    size_filtered_playlists = [
        playlist for playlist in filtered_playlists
        if playlist['tracks']['total'] <= MAX_PLAYLIST_TRACKS
    ]
    
    # Step 3: Sort by follower count and take top FINAL_PLAYLISTS_LIMIT playlists
    sorted_playlists = sorted(
        size_filtered_playlists,
        key=lambda x: x.get('followers', {}).get('total', 0),
        reverse=True
    )[:FINAL_PLAYLISTS_LIMIT]
    
    return sorted_playlists

def get_tracks_batch(sp: spotipy.Spotify, playlist: Dict) -> Tuple[Dict, List[Dict]]:
    """Get all tracks from a playlist efficiently using fields filter and pagination"""
    fields = 'items(track(name,id,artists(name),album(release_date,images),popularity)),next'
    tracks = []
    results = sp.playlist_tracks(
        playlist['id'],
        fields=fields,
        market=MARKET
    )
    
    # Get all tracks using pagination
    tracks.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    
    return playlist, tracks

def process_playlist_tracks(args: Tuple[spotipy.Spotify, Dict]) -> Tuple[Dict, List[Dict]]:
    """Helper function to process playlist tracks in parallel"""
    sp, playlist = args
    return get_tracks_batch(sp, playlist)

def get_viral_playlists_tracks(sp: spotipy.Spotify) -> Dict[str, int]:
    """Get tracks from viral playlists and count their appearances"""
    viral_tracks = defaultdict(int)
    max_appearances = 0

    try:
        with open('viral.csv', 'r') as f:
            for line in f:
                try:
                    _, playlist_url = line.strip().split(',')
                    playlist_id = playlist_url.split('/')[-1]
                    
                    # Get tracks from the viral playlist
                    results = sp.playlist_tracks(
                        playlist_id,
                        fields='items(track(name,id,artists(name)))',
                        market=MARKET
                    )
                    
                    for item in results['items']:
                        if item['track']:
                            track = item['track']
                            artists = ', '.join([artist['name'] for artist in track['artists']])
                            track_key = f"{track['name']}|||{artists}"
                            viral_tracks[track_key] += 1
                            max_appearances = max(max_appearances, viral_tracks[track_key])
                except Exception as e:
                    continue
    except Exception as e:
        st.warning(f"Could not read viral playlists: {str(e)}")
        return {}

    # Normalize scores to 0-100
    if max_appearances > 0:
        return {k: (v / max_appearances) * 100 for k, v in viral_tracks.items()}
    return {}

def get_top_tracks_from_multiple_playlists(sp: spotipy.Spotify, playlists: List[Dict]) -> List[Dict]:
    """Get top tracks from multiple playlists using parallel processing"""
    # Get viral tracks only if virality is enabled
    viral_tracks = get_viral_playlists_tracks(sp) if use_virality else {}
    
    track_info = defaultdict(lambda: {
        'count': 0, 
        'popularity': 0, 
        'name': '', 
        'artist': '', 
        'release_date': '',
        'id': '',
        'artwork_url': '',
        'virality': 0
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
                    'count': track_info[track_key]['count'] + 1,
                    'artwork_url': artwork_url
                })
    
    # Filter out tracks that appear in only one playlist
    track_info = {k: v for k, v in track_info.items() if v['count'] > 1}
    
    if not track_info:
        return []
    
    # Calculate scores
    max_count = max(info['count'] for info in track_info.values())
    tracks_list = []
    
    for info in track_info.values():
        if all(key in info for key in ['name', 'artist', 'popularity', 'release_date']):
            popularity_score = info['popularity']
            freshness_score = calculate_freshness(info['release_date'])
            appearance_score = calculate_appearance_score(info['count'], max_count)
            
            # Get virality score only if enabled
            virality_score = 0
            if use_virality:
                track_key = f"{info['name']}|||{info['artist']}"
                virality_score = viral_tracks.get(track_key, 0)
            
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
                'url': f"spotify:track:{info['id']}"
            }
            
            # Only add virality if enabled
            if use_virality:
                track_data['virality'] = virality_score
                
            tracks_list.append(track_data)
    
    return sorted(tracks_list, key=lambda x: x['total_score'], reverse=True)[:TOP_TRACKS_LIMIT]

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
    """Helper function to process album details in parallel"""
    sp, batch = args
    album_details = {}
    
    tracks = sp.tracks(batch)['tracks']
    for track in tracks:
        if track and track['album']:
            album_id = track['album']['id']
            album = sp.album(album_id)
            album_details[track['id']] = {
                'label': album.get('label', 'Unknown')
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

st.set_page_config(page_title="Search Tracks - Top Songs Finder", page_icon="🔍", layout="wide")
st.title("🔍 Search Tracks")
st.write("Find the most popular tracks across user-created playlists!")

# Initialize Spotify client
sp = setup_spotify()

# Sidebar for configuration
st.sidebar.title("Settings")

# Basic settings
st.sidebar.subheader("Basic Settings")

# Define common markets with their names
AVAILABLE_MARKETS = {
    'US': 'United States',
    'AR': 'Argentina',
    'AU': 'Australia',
    'BR': 'Brazil',
    'CA': 'Canada',
    'CL': 'Chile',
    'CO': 'Colombia',
    'DE': 'Germany',
    'EC': 'Ecuador',
    'ES': 'Spain',
    'FR': 'France',
    'GB': 'United Kingdom',
    'ID': 'Indonesia',
    'IN': 'India',
    'IT': 'Italy',
    'JP': 'Japan',
    'KR': 'South Korea',
    'KZ': 'Kazakhstan',
    'MY': 'Malaysia',
    'MX': 'Mexico',
    'NL': 'Netherlands',
    'PE': 'Peru',
    'PH': 'Philippines',
    'PL': 'Poland',
    'TH': 'Thailand',
    'TR': 'Turkey',
    'TW': 'Taiwan',
    'UA': 'Ukraine',
    'VN': 'Vietnam'
}

selected_markets = st.sidebar.multiselect(
    "Markets",
    options=list(AVAILABLE_MARKETS.keys()),
    default=['US'],
    format_func=lambda x: f"{x} - {AVAILABLE_MARKETS[x]}",
    help="Select markets to search playlists in"
)

# If no markets selected, default to US
if not selected_markets:
    selected_markets = ['US']

# Add checkbox for label information
show_label_info = st.sidebar.checkbox(
    "Show Label Information",
    value=False,
    help="Display record label information for each track"
)

# Add checkbox for virality feature
use_virality = st.sidebar.checkbox(
    "Use Virality Score",
    value=False,
    help="Include virality score from Spotify's Top 50 Viral playlists in ranking"
)

st.sidebar.markdown("#")  

# Scoring weights
st.sidebar.subheader("Scoring Weights")
weight_popularity = st.sidebar.slider("Popularity Weight", 0.0, 1.0, WEIGHT_POPULARITY, 0.05)
weight_freshness = st.sidebar.slider("Freshness Weight", 0.0, 1.0, WEIGHT_FRESHNESS, 0.05)
weight_appearances = st.sidebar.slider("Appearances Weight", 0.0, 1.0, WEIGHT_APPEARANCES, 0.05)
if use_virality:
    weight_virality = st.sidebar.slider("Virality Weight", 0.0, 1.0, WEIGHT_VIRALITY, 0.05)
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

# Playlist filtering settings (moved to bottom)
st.sidebar.subheader("Playlist Filtering")
max_playlist_tracks = st.sidebar.slider(
    "Max tracks per playlist",
    min_value=50,
    max_value=500,
    value=MAX_PLAYLIST_TRACKS,
    step=50,
    help="Playlists with more tracks than this will be filtered out"
)

final_playlists = st.sidebar.slider(
    "Number of playlists to analyze",
    min_value=5,
    max_value=50,
    value=FINAL_PLAYLISTS_LIMIT,
    step=5,
    help="Number of top playlists to analyze after filtering"
)

top_tracks_limit = st.sidebar.slider(
    "Number of top tracks to show",
    min_value=10,
    max_value=100,
    value=TOP_TRACKS_LIMIT,
    step=10,
    help="Number of top tracks to display in results"
)

st.sidebar.markdown("#")  

# Threshold settings
st.sidebar.subheader("Threshold Settings")

# Add release date threshold
min_release_date = st.sidebar.date_input(
    "Minimum Release Date",
    value=None,
    help="Filter out tracks released before this date (leave empty to include all)"
)

popularity_range = st.sidebar.slider(
    "Popularity Range",
    min_value=0,
    max_value=100,
    value=(0, 100),
    step=5,
    help="Filter tracks within this popularity range"
)

min_freshness = st.sidebar.slider(
    "Minimum Freshness",
    min_value=0,
    max_value=100,
    value=0,
    step=5,
    help="Filter out tracks below this freshness score"
)

min_playlists = st.sidebar.slider(
    "Minimum Playlist Appearances",
    min_value=1,
    max_value=20,
    value=2,
    step=1,
    help="Filter out tracks that appear in fewer playlists"
)

freshness_days = st.sidebar.slider(
    "Freshness Days",
    min_value=30,
    max_value=730,
    value=FRESHNESS_DAYS,
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

# Create input field for keyword
search_col1, search_col2 = st.columns([4, 1])  # Ratio 4:1 for input:button

with search_col1:
    keyword = st.text_input("Enter keywords to search for playlists (separate by comma):", "", label_visibility="collapsed")

with search_col2:
    search_button = st.button("Search", type="primary", use_container_width=True)

if search_button and keyword:
    # Update global variables based on user settings
    MAX_PLAYLIST_TRACKS = max_playlist_tracks
    FINAL_PLAYLISTS_LIMIT = final_playlists
    TOP_TRACKS_LIMIT = top_tracks_limit
    WEIGHT_POPULARITY = weight_popularity
    WEIGHT_FRESHNESS = weight_freshness
    WEIGHT_APPEARANCES = weight_appearances
    WEIGHT_VIRALITY = weight_virality
    MARKET = selected_markets[0]
    FRESHNESS_DAYS = freshness_days
    
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
    
    # Apply threshold filters
    filter_conditions = [
        (df['popularity'] >= popularity_range[0]),
        (df['popularity'] <= popularity_range[1]),
        (df['freshness'] >= min_freshness),
        (df['occurrence'] >= min_playlists)
    ]

    if min_release_date:
        # Convert release_date column to datetime, handling different date formats
        df['release_date'] = pd.to_datetime(df['release_date'].apply(
            lambda x: f"{x}-01-01" if len(x) == 4 else (f"{x}-01" if len(x) == 7 else x)
        ))
        filter_conditions.append(df['release_date'] >= pd.Timestamp(min_release_date))

    if use_virality:
        filter_conditions.extend([
            (df['virality'] >= virality_range[0]),
            (df['virality'] <= virality_range[1])
        ])
    
    df = df[pd.concat(filter_conditions, axis=1).all(axis=1)]
    
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
    
    # Reorder columns and rename them for better display
    columns = ['artwork_url', 'name', 'artist']
    if show_label_info:
        columns.append('Label')
    columns.extend(['total_score', 'popularity', 'freshness', 'occurrence'])
    if use_virality:
        columns.append('virality')
    columns.extend(['release_date', 'url'])
    
    df = df[columns]
    column_names = {
        'artwork_url': 'Artwork',
        'name': 'Track Name',
        'artist': 'Artists',
        'total_score': 'Total Score',
        'popularity': 'Popularity',
        'freshness': 'Freshness',
        'occurrence': 'Playlists',
        'virality': 'Virality',
        'release_date': 'Release Date',
        'url': 'URL'
    }
    df.columns = [column_names.get(col, col) for col in df.columns]
    
    # Round numerical columns
    df['Total Score'] = df['Total Score'].round(1)
    df['Freshness'] = df['Freshness'].round(1)
    
    # Display results
    st.subheader(f"Top {len(df)} Tracks")
    
    # Reset index to start from 1
    df.index = range(1, len(df) + 1)
    
    column_config = {
        "Artwork": st.column_config.ImageColumn("Artwork", width="small"),
        "URL": st.column_config.LinkColumn(
            "Open in Spotify",
            display_text="Open in App",
            help="Click to open in Spotify desktop/mobile app"
        ),
        "Total Score": st.column_config.NumberColumn("Total Score", format="%.1f"),
        "Freshness": st.column_config.NumberColumn("Freshness", format="%.1f"),
        "Popularity": st.column_config.NumberColumn("Popularity", format="%.0f"),
        "Virality": st.column_config.NumberColumn("Virality", format="%.1f"),
        "Artists": st.column_config.TextColumn("Artists", width="medium")
    }
    
    if show_label_info:
        column_config.update({
            "Label": st.column_config.TextColumn("Label", width="medium")
        })
    
    st.dataframe(
        df,
        use_container_width=True,
        column_config=column_config
    )
    
    # Create an expander for analytics
    with st.expander("View Analytics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Popularity vs. Freshness scatter plot
            scatter_color = "Virality" if use_virality else "Total Score"
            scatter_title = "Popularity vs. Freshness (color = Virality)" if use_virality else "Popularity vs. Freshness (color = Total Score)"
            
            fig_scatter = px.scatter(
                df,
                x="Popularity",
                y="Freshness",
                color=scatter_color,
                hover_data=["Track Name", "Artists"],
                title=scatter_title
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            if use_virality:
                # Virality distribution histogram
                fig_virality = px.histogram(
                    df,
                    x="Virality",
                    title="Distribution of Virality Scores",
                    nbins=20
                )
                st.plotly_chart(fig_virality, use_container_width=True)
            else:
                # Show artist distribution when virality is disabled
                all_artists = df['Artists'].str.split(', ').explode()
                artist_counts = all_artists.value_counts().head(10)
                
                fig_artists = px.bar(
                    x=artist_counts.index,
                    y=artist_counts.values,
                    title="Top 10 Artists",
                    labels={'x': 'Artist', 'y': 'Number of Tracks'}
                )
                st.plotly_chart(fig_artists, use_container_width=True)