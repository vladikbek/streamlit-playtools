import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import List, Dict, Tuple
from collections import defaultdict
import concurrent.futures
import pandas as pd
import streamlit as st
from app.config import VIRAL_PLAYLISTS, MAX_WORKERS, BATCH_SIZE
import plotly.express as px

st.title(":material/chart_data: Viral Tracks")
st.write("Find the most viral tracks across Spotify's Viral 50 playlists worldwide!")

def setup_spotify():
    """Initialize Spotify client with credentials from .env file"""
    load_dotenv()
    auth_manager = SpotifyClientCredentials()
    return spotipy.Spotify(auth_manager=auth_manager)

def process_viral_playlist(args: Tuple[spotipy.Spotify, str, str]) -> Tuple[str, str, List[Dict]]:
    """Process a single viral playlist"""
    sp, playlist_uri, country_code = args
    try:
        playlist_id = playlist_uri.split(':')[-1]
        results = sp.playlist_tracks(
            playlist_id,
            fields='items(track(name,id,artists(name),album(id,release_date,images),popularity))',
            market='US'
        )
        return playlist_uri, country_code, results['items']
    except Exception as e:
        st.warning(f"Could not fetch tracks from viral playlist {playlist_uri}: {str(e)}")
        return playlist_uri, country_code, []

def process_album_details(args: Tuple[spotipy.Spotify, List[str]]) -> Dict[str, Dict]:
    """Process album details in parallel using Get Several Albums endpoint"""
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
    
    # Get unique album IDs
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

# Initialize Spotify client
sp = setup_spotify()

# Sidebar settings
st.sidebar.title("Settings")

# Add checkbox for including GLOBAL playlist
include_global = st.sidebar.checkbox(
    "Include GLOBAL",
    value=False,
    help="Include tracks from Global Viral 50 playlist in results"
)

# Add checkbox for label information
show_label_info = st.sidebar.checkbox(
    "Show Record Label",
    value=False,
    help="Display record label information for each track"
)

# Add minimum occurrences filter
min_occurrences = st.sidebar.slider(
    "Minimum Viral Playlists",
    min_value=1,
    max_value=10,
    value=2,
    help="Show tracks that appear in at least this many viral playlists"
)

# Add search button
search_button = st.button("Search Viral Playlists", type="primary", use_container_width=True)

if search_button:
    with st.spinner("Searching viral playlists worldwide..."):
        viral_data = defaultdict(lambda: {'count': 0, 'countries': set(), 'track_info': None})
        
        # Process playlists in parallel
        playlist_args = [
            (sp, playlist_uri, country_code) 
            for playlist_uri, country_code in VIRAL_PLAYLISTS 
            if include_global or country_code != 'GLOBAL'
        ]
        total_playlists = len(playlist_args)
        
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
                            
                            # Store track info if not already stored
                            if not viral_data[track_key]['track_info']:
                                viral_data[track_key]['track_info'] = {
                                    'artwork_url': track['album']['images'][-1]['url'] if track['album']['images'] else '',
                                    'name': track['name'],
                                    'artist': [artist['name'] for artist in track['artists']],
                                    'popularity': track.get('popularity', 0),
                                    'release_date': track['album']['release_date'],
                                    'id': track['id'],
                                    'album_id': track['album']['id']
                                }
                except Exception as e:
                    st.warning(f"Error processing viral playlist results: {str(e)}")
                    continue
        
        # Convert to DataFrame
        tracks_list = []
        for track_key, info in viral_data.items():
            if info['count'] >= min_occurrences:
                track_data = info['track_info'].copy()
                track_data.update({
                    'viral_count': info['count'],
                    'viral_countries': sorted(info['countries']),
                    'url': f"spotify:track:{track_data['id']}"
                })
                tracks_list.append(track_data)
        
        if not tracks_list:
            st.error("No tracks found matching the criteria.")
            st.stop()
            
        df = pd.DataFrame(tracks_list)
        
        if df.empty:
            st.error("No tracks found after applying filters.")
            st.stop()
        
        # Get label information if checkbox is checked
        if show_label_info:
            with st.spinner("Fetching label information..."):
                track_ids = [url.split(':')[-1] for url in df['url']]
                album_details = get_album_details_batch(sp, track_ids)
                df['Label'] = df['url'].apply(lambda x: album_details.get(x.split(':')[-1], {}).get('label', 'Unknown'))
        
        # Sort by viral count
        df = df.sort_values('viral_count', ascending=False)
        
        # Add Stats URL
        df['Stats'] = df['url'].apply(lambda x: f"https://www.mystreamcount.com/track/{x.split(':')[-1]}")
        
        # Reset index to start from 1
        df.index = range(1, len(df) + 1)
        
        # Configure columns
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
            "artist": st.column_config.ListColumn(
                "Artists",
                help="Track artists",
                width="medium"
            ),
            "popularity": st.column_config.NumberColumn(
                "Popularity",
                help="Spotify's popularity score (0-100)",
                width="small",
                format="%d"
            )
        }
        
        if show_label_info:
            column_config["Label"] = st.column_config.TextColumn(
                "Label",
                help="Record label that released the track",
                width="medium"
            )
            
        column_config.update({
            "viral_count": st.column_config.NumberColumn(
                "Viral Count",
                help="Number of viral playlists the track appears in",
                width="small"
            ),
            "viral_countries": st.column_config.ListColumn(
                "Viral Countries",
                help="Countries where the track appears in viral playlists",
                width="medium"
            ),
            "release_date": st.column_config.DateColumn(
                "Release Date",
                help="Track's release date"
            ),
            "url": st.column_config.LinkColumn(
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
            )
        })
        
        # Display results
        st.subheader(f"Found {len(df)} Viral Tracks")
        
        # Select and reorder columns
        display_columns = ['artwork_url', 'name', 'artist', 'popularity']
        if show_label_info:
            display_columns.append('Label')
        display_columns.extend(['viral_count', 'viral_countries', 'release_date', 'url', 'Stats'])
        
        st.dataframe(
            df[display_columns],
            use_container_width=True,
            column_config=column_config,
            hide_index=False
        )
        
        # Create an expander for analytics
        with st.expander("View Analytics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Get top 10 artists
                all_artists = [artist for artists_list in df['artist'] for artist in artists_list]
                artist_counts = pd.Series(all_artists).value_counts().head(10)
                
                fig_artists = px.bar(
                    x=artist_counts.index,
                    y=artist_counts.values,
                    title="Top 10 Artists",
                    labels={'x': 'Artist', 'y': 'Number of Tracks'}
                )
                st.plotly_chart(fig_artists, use_container_width=True)
            
            with col2:
                if show_label_info:
                    # Get top 10 labels
                    label_counts = df['Label'].value_counts().head(10)
                    
                    fig_labels = px.bar(
                        x=label_counts.index,
                        y=label_counts.values,
                        title="Top 10 Record Labels",
                        labels={'x': 'Label', 'y': 'Number of Tracks'}
                    )
                    st.plotly_chart(fig_labels, use_container_width=True)
                else:
                    st.info("Enable 'Show Record Label' in settings to view label analytics.") 
