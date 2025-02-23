import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date
import concurrent.futures
from config import AVAILABLE_MARKETS, BATCH_SIZE, MAX_WORKERS, MAX_RESULTS

# Configuration variables
def setup_spotify():
    """Initialize Spotify client with credentials from .env file"""
    load_dotenv()
    auth_manager = SpotifyClientCredentials()
    return spotipy.Spotify(auth_manager=auth_manager)

def search_tracks_by_label(sp: spotipy.Spotify, label: str, market: str, limit: int = 50, current_year_only: bool = False) -> list:
    """Search for tracks by record label"""
    tracks = []
    offset = 0
    
    # Use exact match with quotes for more precise results
    query = f'label:"{label}"'
    if current_year_only:
        query += f' year:{datetime.now().year}'
    
    while True:
        results = sp.search(q=query, type='track', limit=limit, offset=offset, market=market)
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

# Page configuration
st.set_page_config(page_title="Search by Label - Top Songs Finder", page_icon="🏢", layout="wide")
st.title("🏢 Search by Label")
st.write("Find tracks by record label and see their popularity!")

# Initialize Spotify client
sp = setup_spotify()

# Sidebar for configuration
st.sidebar.title("Settings")

# Market selection
selected_market = st.sidebar.selectbox(
    "Market",
    options=list(AVAILABLE_MARKETS.keys()),
    format_func=lambda x: f"{x} - {AVAILABLE_MARKETS[x]}",
    index=0
)

# Filtering options
st.sidebar.subheader("Filtering Options")

# Add current year filter
show_current_year = st.sidebar.checkbox(
    "Show Current Year Only",
    value=False,
    help=f"Show only tracks released in {datetime.now().year}"
)

# Add ISRC filter
isrc_filter = st.sidebar.text_input(
    "Filter by ISRC",
    value="",
    help="Enter one or more ISRCs (comma-separated). Will match any track containing the entered text."
)

min_popularity = st.sidebar.slider(
    "Minimum Popularity",
    min_value=0,
    max_value=100,
    value=0,
    help="Filter tracks by minimum popularity score"
)

# Date range filter
min_release_date = st.sidebar.date_input(
    "Minimum Release Date",
    value=None,
    help="Filter tracks released after this date (leave empty for no filter)"
)

# Create search interface
search_col1, search_col2 = st.columns([4, 1])

with search_col1:
    search_label = st.text_input(
        "Enter record label name:",
        "",
        help="Enter the exact name of the record label",
        label_visibility="collapsed"
    )

with search_col2:
    search_button = st.button("Search", type="primary", use_container_width=True)

# Global variable to store the search label
if 'search_label' not in st.session_state:
    st.session_state.search_label = None

if search_button and search_label:
    st.session_state.search_label = search_label
    
    with st.spinner(f"Searching for tracks from {search_label}..."):
        # Search for tracks with current_year parameter
        tracks = search_tracks_by_label(sp, search_label, selected_market, current_year_only=show_current_year)
        
        if not tracks:
            st.warning(f"No tracks found for label: {search_label}")
            st.stop()
            
        # Process tracks in parallel
        batches = [tracks[i:i + BATCH_SIZE] for i in range(0, len(tracks), BATCH_SIZE)]
        processed_tracks = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_tracks_batch, (sp, batch)) for batch in batches]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    processed_tracks.extend(batch_results)
                except Exception as e:
                    st.warning(f"Error processing some tracks: {str(e)}")
                    continue
        
        if not processed_tracks:
            st.warning(f"No tracks found matching the exact label: {search_label}")
            st.stop()
            
        # Convert to DataFrame
        df = pd.DataFrame(processed_tracks)
        
        # Apply filters
        if min_popularity > 0:
            df = df[df['popularity'] >= min_popularity]
            
        if min_release_date:
            df['release_date'] = pd.to_datetime(df['release_date'].apply(
                lambda x: f"{x}-01-01" if len(x) == 4 else (f"{x}-01" if len(x) == 7 else x)
            ))
            df = df[df['release_date'] >= pd.Timestamp(min_release_date)]
            
        # Apply ISRC filter if provided
        if isrc_filter:
            # Split by comma and strip whitespace
            isrc_list = [isrc.strip().upper() for isrc in isrc_filter.split(',') if isrc.strip()]
            if isrc_list:
                # Create a mask that matches if any of the ISRC patterns are found
                isrc_mask = df['isrc'].str.contains('|'.join(isrc_list), case=False, na=False)
                df = df[isrc_mask]
        
        # Sort by popularity
        df = df.sort_values('popularity', ascending=False)
        
        if df.empty:
            st.error("No tracks found matching the filter criteria.")
            st.stop()
        
        # Reset index to start from 1
        df.index = range(1, len(df) + 1)
        
        # Display results
        st.subheader(f"Top {len(df)} Tracks from {search_label}")
        
        # Configure columns for display
        column_config = {
            "artwork_url": st.column_config.ImageColumn("Artwork", width="small"),
            "name": st.column_config.TextColumn("Track Name", width="medium"),
            "artists": st.column_config.ListColumn(
                "Artists",
                help="Artists who performed on this track",
                width="medium"
            ),
            "popularity": st.column_config.NumberColumn("Popularity", format="%d"),
            "isrc": st.column_config.TextColumn(
                "ISRC",
                help="International Standard Recording Code",
                width="small"
            ),
            "release_date": st.column_config.DateColumn("Release Date"),
            "url": st.column_config.LinkColumn(
                "▶️ Play",
                display_text="Play",
                help="Click to open in Spotify desktop/mobile app"
            )
        }
        
        st.dataframe(
            df,
            use_container_width=True,
            column_config=column_config
        )
        
        # Analytics section
        with st.expander("View Analytics", expanded=False):
            # Calculate average popularity excluding zeros
            non_zero_popularity = df[df['popularity'] > 0]['popularity']
            avg_popularity = non_zero_popularity.mean()
            st.metric(
                "Average Track Popularity",
                f"{avg_popularity:.1f}",
                help="The average popularity score across all tracks (excluding tracks with 0 popularity)"
            )
            
            # Create three columns for graphs
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                # Popularity distribution with 10-point bins and colors
                fig_popularity = px.histogram(
                    df,
                    x="popularity",
                    title="Distribution of Track Popularity",
                    nbins=10,  # Split by 10
                    color_discrete_sequence=['#1DB954'],  # Spotify green
                    opacity=0.8
                )
                
                # Customize layout
                fig_popularity.update_layout(
                    xaxis_title="Popularity Score",
                    yaxis_title="Number of Tracks",
                    bargap=0.1,  # Add some gap between bars
                )
                
                # Add vertical line for average
                fig_popularity.add_vline(
                    x=avg_popularity,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Average: {avg_popularity:.1f}",
                    annotation_position="top"
                )
                
                st.plotly_chart(fig_popularity, use_container_width=True)
            
            # Calculate artist statistics
            artist_popularity = []
            for idx, row in df.iterrows():
                for artist in row['artists']:
                    artist_popularity.append({
                        'artist': artist,
                        'popularity': row['popularity']
                    })
            
            # Convert to DataFrame and calculate statistics
            artist_df = pd.DataFrame(artist_popularity)
            artist_stats = artist_df.groupby('artist').agg({
                'popularity': ['mean', 'count']
            }).reset_index()
            
            # Flatten column names
            artist_stats.columns = ['artist', 'avg_popularity', 'track_count']
            
            with col2:
                # Top artists by number of tracks
                top_by_tracks = artist_stats.nlargest(10, 'track_count')
                fig_tracks = px.bar(
                    top_by_tracks,
                    x='artist',
                    y='track_count',
                    title="Top 10 Artists by Number of Tracks",
                    color='track_count',
                    color_continuous_scale='Viridis',
                    labels={
                        'artist': 'Artist',
                        'track_count': 'Number of Tracks'
                    }
                )
                
                # Customize layout
                fig_tracks.update_layout(
                    xaxis_tickangle=45,
                    showlegend=False,
                    yaxis_title="Number of Tracks"
                )
                
                st.plotly_chart(fig_tracks, use_container_width=True, theme="streamlit")
            
            with col3:
                # Top artists by average popularity (minimum 2 tracks)
                top_by_popularity = artist_stats[artist_stats['track_count'] >= 2].nlargest(10, 'avg_popularity')
                fig_artist_popularity = px.bar(
                    top_by_popularity,
                    x='artist',
                    y='avg_popularity',
                    title="Top 10 Artists by Popularity",
                    color='track_count',
                    color_continuous_scale='Viridis',
                    labels={
                        'artist': 'Artist',
                        'avg_popularity': 'Average Popularity',
                        'track_count': 'Number of Tracks'
                    }
                )
                
                # Customize layout
                fig_artist_popularity.update_layout(
                    xaxis_tickangle=45,
                    showlegend=True,
                    yaxis_title="Average Popularity"
                )
                
                st.plotly_chart(fig_artist_popularity, use_container_width=True)
            
            # Show detailed artist stats in a table
            st.subheader("Artist Statistics")
            artist_stats['avg_popularity'] = artist_stats['avg_popularity'].round(1)
            st.dataframe(
                artist_stats.sort_values('avg_popularity', ascending=False),
                column_config={
                    'artist': st.column_config.TextColumn("Artist"),
                    'avg_popularity': st.column_config.NumberColumn("Average Popularity", format="%.1f"),
                    'track_count': st.column_config.NumberColumn("Number of Tracks")
                },
                hide_index=True,
                use_container_width=True
            ) 