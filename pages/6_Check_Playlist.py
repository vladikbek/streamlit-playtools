import os
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import streamlit as st
import plotly.express as px
import concurrent.futures
from typing import List, Dict, Tuple
from collections import Counter
from config import BATCH_SIZE, MAX_WORKERS
import time

def setup_spotify():
    """Initialize Spotify client with credentials from .env file"""
    load_dotenv()
    auth_manager = SpotifyClientCredentials()
    return spotipy.Spotify(auth_manager=auth_manager)

def extract_playlist_id(input_str: str) -> str:
    """Extract playlist ID from various input formats"""
    if not input_str:
        return ""
    
    # Handle direct playlist ID
    if len(input_str) == 22 and all(c.isalnum() for c in input_str):
        return input_str
        
    # Handle Spotify URI
    if "spotify:playlist:" in input_str:
        return input_str.split("spotify:playlist:")[-1]
        
    # Handle Spotify URL
    if "/playlist/" in input_str:
        return input_str.split("/playlist/")[-1].split("?")[0]
    
    return ""

def process_track_batch(args: Tuple[spotipy.Spotify, List[Dict], int, bool]) -> Tuple[int, List[Dict]]:
    """Process a batch of tracks in parallel while preserving order"""
    sp, tracks_batch, batch_index, show_isrc = args
    processed_tracks = []
    
    # First get track IDs and artist IDs for batch lookups
    track_ids = []
    artist_ids = set()  # Using set to avoid duplicates
    track_to_artists = {}  # Map track IDs to their artist IDs
    
    for track_item in tracks_batch:
        if not track_item['track']:
            continue
        track = track_item['track']
        track_ids.append(track['id'])
        track_artists = []
        for artist in track['artists']:
            artist_ids.add(artist['id'])
            track_artists.append(artist['id'])
        track_to_artists[track['id']] = track_artists
    
    # Get album details for the batch (for labels)
    album_details = {}
    if track_ids:
        # Split into sub-batches of 20 for Spotify API limit
        for i in range(0, len(track_ids), 20):
            sub_batch = track_ids[i:i + 20]
            tracks_info = sp.tracks(sub_batch)['tracks']
            
            # Collect album IDs
            album_ids = []
            track_to_album = {}  # Map track IDs to their album IDs
            for track in tracks_info:
                if track and track['album']:
                    album_id = track['album']['id']
                    album_ids.append(album_id)
                    track_to_album[track['id']] = album_id
            
            # Get unique album IDs
            unique_album_ids = list(set(album_ids))
            
            # Fetch albums in batches of 20
            for j in range(0, len(unique_album_ids), 20):
                album_batch = unique_album_ids[j:j + 20]
                albums_response = sp.albums(album_batch)
                
                # Create mapping of album ID to label
                for album in albums_response['albums']:
                    if album:
                        album_details[album['id']] = {
                            'label': album.get('label', 'Unknown')
                        }
    
    # Get artist details (for genres)
    artist_details = {}
    artist_list = list(artist_ids)
    for i in range(0, len(artist_list), 50):  # Spotify allows up to 50 artists per request
        artist_batch = artist_list[i:i + 50]
        artists_response = sp.artists(artist_batch)
        
        # Create mapping of artist ID to genres
        for artist in artists_response['artists']:
            if artist:
                artist_details[artist['id']] = {
                    'genres': artist.get('genres', [])
                }
    
    # Process each track in the batch
    for track_item in tracks_batch:
        if not track_item['track']:
            continue
            
        track = track_item['track']
        album_id = track['album']['id']
        album_info = album_details.get(album_id, {'label': 'Unknown'})
        
        # Collect genres from all artists of the track
        track_genres = set()
        for artist_id in track_to_artists.get(track['id'], []):
            artist_info = artist_details.get(artist_id, {})
            track_genres.update(artist_info.get('genres', []))
        
        processed_track = {
            'artwork_url': track['album']['images'][-1]['url'] if track['album']['images'] else '',
            'name': track['name'],
            'artists': [artist['name'] for artist in track['artists']],
            'label': album_info['label'],
            'genres': sorted(list(track_genres)),  # Convert set back to sorted list
            'popularity': track['popularity'],
            'date_added': track_item['added_at'],
            'release_date': track['album']['release_date'],
            'url': f"spotify:track:{track['id']}",
            'stats': f"https://www.mystreamcount.com/track/{track['id']}"
        }
        
        # Add ISRC if requested
        if show_isrc:
            processed_track['isrc'] = track.get('external_ids', {}).get('isrc', 'N/A')
        
        processed_tracks.append(processed_track)
    
    return batch_index, processed_tracks

def get_playlist_tracks(sp: spotipy.Spotify, playlist_id: str, show_isrc: bool = False) -> List[Dict]:
    """Get all tracks from a playlist with parallel processing"""
    # First, get all tracks from the playlist
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    
    if not tracks:
        return []
    
    # Split tracks into batches for parallel processing
    track_batches = [tracks[i:i + BATCH_SIZE] for i in range(0, len(tracks), BATCH_SIZE)]
    processed_tracks = []
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create batch arguments with index for ordering
        batch_args = [(sp, batch, i, show_isrc) for i, batch in enumerate(track_batches)]
        futures = [executor.submit(process_track_batch, args) for args in batch_args]
        
        # Collect results and maintain order
        results = []
        for future in concurrent.futures.as_completed(futures):
            batch_index, batch_tracks = future.result()
            results.append((batch_index, batch_tracks))
        
        # Sort by batch index and combine results
        results.sort(key=lambda x: x[0])
        for _, batch_tracks in results:
            processed_tracks.extend(batch_tracks)
    
    return processed_tracks

st.set_page_config(page_title="Check Playlist - Top Songs Finder", page_icon="📋", layout="wide")
st.title("📋 Check Playlist")
st.write("Analyze all tracks in a Spotify playlist!")

# Initialize Spotify client
sp = setup_spotify()

# Sidebar settings
st.sidebar.title("Settings")

# Add checkbox for ISRC display
show_isrc = st.sidebar.checkbox(
    "Show ISRC codes",
    value=False,
    help="Display International Standard Recording Code for each track"
)

# Get playlist ID from URL params
playlist_id_param = st.query_params.get("playlist", "")

# Create input field for playlist URL/URI with the same design as Search Tracks
search_col1, search_col2 = st.columns([4, 1])  # Ratio 4:1 for input:button

with search_col1:
    playlist_input = st.text_input(
        "Enter Spotify playlist URL, URI, or ID:",
        value=playlist_id_param,  # Use the playlist ID from URL params
        help="Example: https://open.spotify.com/playlist/xxxxx or spotify:playlist:xxxxx or just the playlist ID",
        label_visibility="collapsed"
    )

with search_col2:
    search_button = st.button("Analyze", type="primary", use_container_width=True)

# Process playlist if ID is in URL params or input field
playlist_to_process = playlist_input or playlist_id_param
if playlist_to_process:
    # Extract playlist ID from input
    playlist_id = extract_playlist_id(playlist_to_process)
    
    if not playlist_id:
        st.error("Invalid playlist URL, URI, or ID. Please enter a valid Spotify playlist identifier.")
        st.stop()
    
    # Update URL params with playlist ID if it's not already set
    if st.query_params.get("playlist") != playlist_id:
        st.query_params["playlist"] = playlist_id
    
    try:
        # Get playlist details
        playlist = sp.playlist(playlist_id, fields="name,owner,tracks(total)")
        st.subheader(f"{playlist['name']} by {playlist['owner']['display_name']}")
        
        with st.spinner(f"Analyzing {playlist['tracks']['total']} tracks..."):
            tracks = get_playlist_tracks(sp, playlist_id, show_isrc)
            
            if not tracks:
                st.error("No tracks found in the playlist.")
                st.stop()
            
            # Convert to DataFrame
            df = pd.DataFrame(tracks)
            
            # Reorder columns to place ISRC after release_date if show_isrc is enabled
            if show_isrc and 'isrc' in df.columns:
                columns = list(df.columns)
                columns.remove('isrc')
                # Insert ISRC after release_date
                release_date_idx = columns.index('release_date')
                columns.insert(release_date_idx + 1, 'isrc')
                df = df[columns]
            
            # Convert date columns to datetime with better error handling
            # Handle date_added column
            df['date_added'] = pd.to_datetime(df['date_added'], utc=True)
            
            # Handle release_date column with various formats
            def parse_release_date(date_str):
                if pd.isna(date_str):
                    return pd.NaT
                try:
                    # Handle year-only format
                    if len(str(date_str)) == 4:
                        return pd.to_datetime(f"{date_str}-01-01")
                    # Handle year-month format
                    elif len(str(date_str)) == 7:
                        return pd.to_datetime(f"{date_str}-01")
                    # Handle full date format
                    else:
                        return pd.to_datetime(date_str)
                except:
                    return pd.NaT

            df['release_date'] = df['release_date'].apply(parse_release_date)
            
            # Configure columns for display
            column_config = {
                "artwork_url": st.column_config.ImageColumn(
                    "Artwork",
                    width="small",
                    help="Track's album artwork"
                ),
                "name": st.column_config.TextColumn(
                    "Track Name",
                    help="Name of the track",
                    width="medium"
                ),
                "artists": st.column_config.ListColumn(
                    "Artists",
                    help="Artists who performed this track",
                    width="medium"
                ),
                "label": st.column_config.TextColumn(
                    "Label",
                    help="Record label that released the track",
                    width="medium"
                ),
                "genres": st.column_config.ListColumn(
                    "Genres",
                    help="Artist genres",
                    width="medium"
                ),
                "popularity": st.column_config.NumberColumn(
                    "Popularity",
                    help="Spotify's popularity score (0-100)",
                    format="%d"
                ),
                "date_added": st.column_config.DatetimeColumn(
                    "Date Added",
                    help="When the track was added to the playlist",
                    format="D MMM YYYY"
                ),
                "release_date": st.column_config.DateColumn(
                    "Release Date",
                    help="Track's release date",
                    format="D MMM YYYY"
                ),
                "url": st.column_config.LinkColumn(
                    "Play",
                    display_text="▶️ Play",
                    help="Click to open in Spotify desktop/mobile app",
                    width="small"
                ),
                "stats": st.column_config.LinkColumn(
                    "Stats",
                    display_text="📊 Stats",
                    help="Click to view track statistics",
                    width="small"
                )
            }
            
            # Add ISRC column if requested
            if show_isrc:
                column_config["isrc"] = st.column_config.TextColumn(
                    "ISRC",
                    help="International Standard Recording Code"
                )
            
            # Display the tracks table with 1-based indexing
            df.index = range(1, len(df) + 1)
            st.dataframe(
                df,
                use_container_width=True,
                column_config=column_config,
                hide_index=False
            )
            
            # Create analytics section
            with st.expander("View Analytics", expanded=False):

                # Average Popularity
                avg_popularity = df['popularity'].mean()
                st.metric("Average Popularity", f"{avg_popularity:.1f}")

                col1, col2 = st.columns(2)
                
                with col1:
                    # Top Labels pie chart
                    label_counts = df['label'].value_counts().head(5)
                    fig_labels = px.pie(
                        values=label_counts.values,
                        names=label_counts.index,
                        title="Top 5 Record Labels"
                    )
                    fig_labels.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_labels, use_container_width=True)
                
                with col2:
                    # Popularity Distribution
                    fig_popularity = px.histogram(
                        df,
                        x="popularity",
                        nbins=20,
                        title="Popularity Distribution"
                    )
                    fig_popularity.update_layout(
                        xaxis_title="Popularity Score",
                        yaxis_title="Number of Tracks"
                    )
                    st.plotly_chart(fig_popularity, use_container_width=True)
                
                # Top Genres (flatten the genres list and count)
                all_genres = [genre for genres in df['genres'] for genre in genres]
                genre_counts = pd.Series(all_genres).value_counts().head(10)
                
                if not genre_counts.empty:
                    fig_genres = px.bar(
                        x=genre_counts.index,
                        y=genre_counts.values,
                        title="Top 10 Genres",
                        labels={'x': 'Genre', 'y': 'Number of Tracks'}
                    )
                    fig_genres.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_genres, use_container_width=True)
                else:
                    st.info("No genre information available for the tracks in this playlist.")
                    
    except Exception as e:
        st.error(f"Error analyzing playlist: {str(e)}")
        st.stop() 