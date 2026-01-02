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

st.title(":material/chart_data: Viral Tracks", anchor=False)
st.caption("Find the most viral tracks across Spotify's Viral 50 playlists worldwide.")

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

with st.container(border=True):
    with st.form("viral_tracks_form", border=False):
        search_button = st.form_submit_button(
            "Search Viral Playlists",
            type="primary",
            icon=":material/search:",
            width="stretch"
        )

    min_occurrences = st.slider(
        "Minimum Viral Playlists",
        min_value=1,
        max_value=10,
        value=2,
        help="Show tracks that appear in at least this many viral playlists"
    )

if search_button:
    with st.spinner("Searching viral playlists worldwide..."):
        viral_data = defaultdict(lambda: {'count': 0, 'countries': set(), 'track_info': None})
        
        # Process playlists in parallel
        playlist_args = [
            (sp, playlist_uri, country_code)
            for playlist_uri, country_code in VIRAL_PLAYLISTS
            if country_code != "GLOBAL"
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
            if info['track_info']:
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
        
        # Always fetch label information
        track_ids = [url.split(':')[-1] for url in df['url']]
        album_details = get_album_details_batch(sp, track_ids)
        df['Label'] = df['url'].apply(lambda x: album_details.get(x.split(':')[-1], {}).get('label', 'Unknown'))
        
        # Sort by viral count
        df = df.sort_values('viral_count', ascending=False)
        
        # Add Stats URL
        df['Stats'] = df['url'].apply(lambda x: f"https://www.mystreamcount.com/track/{x.split(':')[-1]}")
        
        st.session_state["viral_results"] = {"df": df}

viral_results = st.session_state.get("viral_results")
if viral_results:
    df = viral_results["df"]
    filtered_df = df[df['viral_count'] >= min_occurrences].copy()

    if filtered_df.empty:
        st.warning("No tracks found after applying the minimum viral playlists filter.")
        st.stop()

    st.subheader(f"Found {len(filtered_df)} viral tracks", anchor=False)
    st.caption(f"Appearing in at least {min_occurrences} viral playlists")

    primary_color = st.get_option("theme.primaryColor")
    tracks_tab, artists_tab, labels_tab = st.tabs(["Tracks", "Artists", "Labels"])

    # Configure columns
    column_config = {
        "url": st.column_config.LinkColumn(
            "Link",
            display_text=":material/open_in_new:",
            help="Click to open in Spotify desktop/mobile app",
            width="small"
        ),
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
        "Label": st.column_config.TextColumn(
            "Label",
            help="Record label that released the track",
            width="medium"
        ),
        "popularity": st.column_config.NumberColumn(
            "Popularity",
            help="Spotify's popularity score (0-100)",
            width="small",
            format="%d"
        ),
        "viral_count": st.column_config.NumberColumn(
            "Viral Count",
            help="Number of viral playlists the track appears in",
            width="small"
        ),
        "viral_countries": st.column_config.ListColumn(
            "Countries",
            help="Countries where the track appears in viral playlists",
            width="large"
        ),
        "release_date": st.column_config.DateColumn(
            "Release Date",
            help="Track's release date"
        ),
        "Stats": st.column_config.LinkColumn(
            "Stats",
            display_text=":material/query_stats:",
            help="Click to view track statistics",
            width="small"
        )
    }

    with tracks_tab:
        display_df = filtered_df.copy()
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(
            display_df,
            width="stretch",
            height=500,
            column_order=[
                "url",
                "artwork_url",
                "name",
                "artist",
                "Label",
                "popularity",
                "viral_count",
                "viral_countries",
                "release_date",
                "Stats"
            ],
            column_config=column_config,
            hide_index=False
        )

    with artists_tab:
        all_artists = [artist for artists_list in filtered_df['artist'] for artist in artists_list]
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

    with labels_tab:
        label_counts = filtered_df['Label'].value_counts().head(10)

        fig_labels = px.bar(
            x=label_counts.index,
            y=label_counts.values,
            title="Top 10 Record Labels",
            labels={'x': 'Label', 'y': 'Number of Tracks'}
        )
        if primary_color:
            fig_labels.update_traces(marker_color=primary_color)
        st.plotly_chart(fig_labels, width="stretch")
