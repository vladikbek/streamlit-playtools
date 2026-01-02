import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import streamlit as st
import plotly.express as px
import concurrent.futures
from typing import List, Dict, Tuple
from app.config import AVAILABLE_MARKETS, BATCH_SIZE, MAX_WORKERS

def setup_spotify():
    """Initialize Spotify client with credentials from .env file"""
    load_dotenv()
    auth_manager = SpotifyClientCredentials()
    return spotipy.Spotify(auth_manager=auth_manager)

def extract_playlist_id(playlist_url: str) -> str:
    """Extract playlist ID from Spotify URL or URI"""
    if 'spotify:playlist:' in playlist_url:
        return playlist_url.split('spotify:playlist:')[1]
    elif 'open.spotify.com/playlist/' in playlist_url:
        playlist_id = playlist_url.split('playlist/')[1]
        # Remove any query parameters
        return playlist_id.split('?')[0]
    return playlist_url  # Assume it's already an ID

def search_playlists(sp: spotipy.Spotify, keyword: str, market: str, target_playlist_id: str, limit: int = 50) -> dict:
    """Search for playlists matching the keyword and find position of target playlist"""
    results = []
    offset = 0
    total_fetched = 0
    found_position = -1
    
    while total_fetched < limit:
        current_limit = min(50, limit - total_fetched)  # Spotify max is 50 per request
        try:
            response = sp.search(q=keyword, type='playlist', limit=current_limit, market=market, offset=offset)
            items = response['playlists']['items']
            
            if not items:  # No more results
                break
                
            # Check if our target playlist is in the results
            for position, playlist in enumerate(items, start=offset + 1):
                if playlist['id'] == target_playlist_id:
                    found_position = position
                    return {
                        'market': market,
                        'keyword': keyword,
                        'position': found_position,
                        'found': True,
                        'playlist_id': target_playlist_id
                    }
            
            total_fetched += len(items)
            offset += len(items)
            
            if len(items) < current_limit:  # Last page
                break
                
        except Exception as e:
            st.warning(f"Error searching in market {market} for keyword '{keyword}': {str(e)}")
            break
    
    return {
        'market': market,
        'keyword': keyword,
        'position': None,
        'found': False,
        'playlist_id': target_playlist_id
    }

def search_playlist_parallel(args: Tuple[spotipy.Spotify, str, str, str, int]) -> Dict:
    """Search for playlist position with keyword and market in parallel"""
    sp, keyword, market, target_playlist_id, limit = args
    return search_playlists(sp, keyword, market, target_playlist_id, limit)

st.title(":material/analytics: Playlist SEO")
st.write("Analyze your playlist's search position across different markets and keywords!")

# Initialize Spotify client
sp = setup_spotify()

# Get query parameters from URL
playlists_param = st.query_params.get("playlists", "")
keywords_param = st.query_params.get("keywords", "")
markets_param = st.query_params.get("markets", "")

# Parse markets from URL parameter if provided
markets_from_url = []
if markets_param:
    markets_from_url = [m.strip() for m in markets_param.split(',') if m.strip() in AVAILABLE_MARKETS]

# Sidebar for configuration
st.sidebar.title("Settings")

# Market selection
selected_markets = st.sidebar.multiselect(
    "Markets",
    options=list(AVAILABLE_MARKETS.keys()),
    default=markets_from_url,
    format_func=lambda x: f"{x} - {AVAILABLE_MARKETS[x]}",
    help="Select markets to search playlists in (leave empty to search all markets)"
)

# Results limit
results_limit = st.sidebar.slider(
    "Search depth per keyword",
    min_value=50,
    max_value=1000,
    value=100,
    step=50,
    help="How deep to search in results (max 1000)"
)

# Create input fields
col1, col2 = st.columns([1, 1])

with col1:
    playlist_url = st.text_input(
        "Enter Spotify playlist URLs/URIs (separate by comma):",
        value=playlists_param,
        placeholder="spotify:playlist:xxx, https://open.spotify.com/playlist/yyy"
    )

with col2:
    keywords = st.text_input(
        "Enter keywords to analyze (separate by comma):",
        value=keywords_param,
        placeholder="edm playlist, electronic music, dance hits"
    )

# Update URL parameters when input values change
if playlist_url != playlists_param or keywords != keywords_param or ','.join(selected_markets) != markets_param:
    # Only update if there are actual values to avoid cluttering the URL
    query_params = {}
    if playlist_url:
        query_params["playlists"] = playlist_url
    if keywords:
        query_params["keywords"] = keywords
    if selected_markets:
        query_params["markets"] = ','.join(selected_markets)
    
    # Update the URL parameters
    st.query_params.update(query_params)

search_button = st.button("Analyze SEO Position", type="primary", use_container_width=True)

if search_button and playlist_url and keywords:
    # Extract playlist IDs
    playlist_urls = [url.strip() for url in playlist_url.split(',') if url.strip()]
    playlist_ids = [extract_playlist_id(url) for url in playlist_urls]
    
    if not playlist_ids:
        st.error("Please enter at least one valid playlist URL/URI.")
        st.stop()
    
    # Validate playlist IDs and get their names
    playlist_infos = []
    for playlist_id in playlist_ids:
        try:
            playlist_info = sp.playlist(playlist_id, fields='name')
            playlist_infos.append({'id': playlist_id, 'name': playlist_info['name']})
        except Exception as e:
            st.error(f"Invalid playlist URL/URI: {playlist_id}. Please check and try again.")
            st.stop()
    
    # Show playlists being analyzed
    playlists_names = ", ".join([info['name'] for info in playlist_infos])
    st.info(f"Analyzing positions for playlists: {playlists_names}")
    
    # Split and clean keywords
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    
    if not keyword_list:
        st.error("Please enter at least one keyword.")
        st.stop()
    
    # Use selected markets or all markets if none selected
    markets_to_search = selected_markets if selected_markets else list(AVAILABLE_MARKETS.keys())
    
    # Create progress bar
    total_searches = len(keyword_list) * len(markets_to_search) * len(playlist_ids)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prepare search arguments
    search_args = [
        (sp, keyword, market, playlist_id, results_limit)
        for playlist_id in playlist_ids
        for keyword in keyword_list
        for market in markets_to_search
    ]
    
    results = []
    completed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total_searches)) as executor:
        futures = [executor.submit(search_playlist_parallel, args) for args in search_args]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            # Add playlist name to the result
            playlist_id = result['playlist_id'] if 'playlist_id' in result else None
            if playlist_id:
                playlist_name = next((info['name'] for info in playlist_infos if info['id'] == playlist_id), 'Unknown')
                result['playlist_name'] = playlist_name
            results.append(result)
            completed += 1
            progress = completed / total_searches
            progress_bar.progress(progress)
            status_text.text(f"Completed {completed}/{total_searches} searches...")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure positions are integers where found
    df['position'] = df['position'].apply(lambda x: int(x) if pd.notnull(x) else x)
    
    # Display detailed results
    st.subheader("Results")
    
    # Prepare detailed results DataFrame
    detailed_df = df[['playlist_name', 'keyword', 'market', 'position']].copy()
    
    # Create a numeric position column for sorting
    detailed_df['sort_position'] = detailed_df['position'].apply(lambda x: float('inf') if pd.isna(x) else x)
    
    # Convert position to string, handling NaN values
    detailed_df['position'] = detailed_df['position'].apply(lambda x: str(int(x)) if pd.notnull(x) else 'Not found')
    # Add full market names
    detailed_df['market'] = detailed_df['market'].apply(lambda x: f"{x} - {AVAILABLE_MARKETS[x]}")
    
    # Sort by position (found positions first, then "Not found")
    detailed_df = detailed_df.sort_values(['sort_position', 'playlist_name'])
    
    # Drop the sorting column and set final column names
    detailed_df = detailed_df.drop('sort_position', axis=1)
    detailed_df.columns = ['Playlist', 'Keyword', 'Market', 'Position']
    
    st.dataframe(
        detailed_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Playlist": st.column_config.TextColumn(
                "Playlist",
                width="medium"
            )
        }
    )
