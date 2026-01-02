import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date
import concurrent.futures
from app.config import AVAILABLE_MARKETS, BATCH_SIZE, MAX_WORKERS, MAX_RESULTS

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

def search_albums_by_label(sp: spotipy.Spotify, label: str, market: str, limit: int = 50, current_year_only: bool = False) -> list:
    """Search for albums by record label"""
    albums = []
    offset = 0
    
    # Use exact match with quotes for more precise results
    query = f'label:"{label}"'
    if current_year_only:
        query += f' year:{datetime.now().year}'
    
    while True:
        results = sp.search(q=query, type='album', limit=limit, offset=offset, market=market)
        if not results['albums']['items']:
            break
            
        albums.extend(results['albums']['items'])
        offset += limit
        
        # Stop if we've reached the maximum number of albums
        if offset >= MAX_RESULTS:
            break
            
    return albums

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

def process_albums_batch(args):
    """Process a batch of albums"""
    sp, albums_batch = args
    processed_albums = []
    
    for album in albums_batch:
        try:
            # Safely get artist names, skipping any None values
            artists = []
            for artist in album.get('artists', []):
                if artist and isinstance(artist, dict) and artist.get('name'):
                    artists.append(artist['name'])
            
            # Skip album if no valid artists found
            if not artists:
                continue
            
            # Get album images safely
            images = album.get('images', [])
            artwork_url = images[-1]['url'] if images else ''
            
            # Get album type and total tracks
            album_type = album.get('album_type', 'album')
            total_tracks = album.get('total_tracks', 0)
            
            album_data = {
                'artwork_url': artwork_url,
                'name': album.get('name', 'Unknown Album'),
                'artists': artists,
                'album_type': album_type,
                'total_tracks': total_tracks,
                'popularity': album.get('popularity', 0),  # Note: popularity might not be available for all albums
                'release_date': album.get('release_date', ''),
                'url': album.get('uri', '')
            }
            processed_albums.append(album_data)
        except Exception as e:
            continue  # Skip albums that cause errors
            
    return processed_albums

# Page configuration
st.title(":material/library_music: Search by Label")
st.write("Find tracks or albums by record label and see their popularity!")

# Initialize Spotify client
sp = setup_spotify()

# Sidebar for configuration
st.sidebar.title("Settings")

# Search type selection
search_type = st.sidebar.radio(
    "Search Type",
    options=["Tracks", "Albums"],
    index=0,
    help="Choose whether to search for tracks or albums by the label"
)

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

# Add ISRC filter (only for tracks)
isrc_filter = ""
if search_type == "Tracks":
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
    
    search_type_lower = search_type.lower()
    with st.spinner(f"Searching for {search_type_lower} from {search_label}..."):
        if search_type == "Tracks":
            # Search for tracks with current_year parameter
            items = search_tracks_by_label(sp, search_label, selected_market, current_year_only=show_current_year)
            process_function = process_tracks_batch
        else:
            # Search for albums with current_year parameter
            items = search_albums_by_label(sp, search_label, selected_market, current_year_only=show_current_year)
            process_function = process_albums_batch
        
        if not items:
            st.warning(f"No {search_type_lower} found for label: {search_label}")
            st.stop()
            
        # Process items in parallel
        batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
        processed_items = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_function, (sp, batch)) for batch in batches]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    processed_items.extend(batch_results)
                except Exception as e:
                    st.warning(f"Error processing some {search_type_lower}: {str(e)}")
                    continue
        
        if not processed_items:
            st.warning(f"No {search_type_lower} found matching the exact label: {search_label}")
            st.stop()
            
        # Convert to DataFrame
        df = pd.DataFrame(processed_items)
        
        # Apply filters
        if min_popularity > 0:
            df = df[df['popularity'] >= min_popularity]
            
        if min_release_date:
            df['release_date'] = pd.to_datetime(df['release_date'].apply(
                lambda x: f"{x}-01-01" if len(x) == 4 else (f"{x}-01" if len(x) == 7 else x)
            ))
            df = df[df['release_date'] >= pd.Timestamp(min_release_date)]
            
        # Apply ISRC filter if provided (only for tracks)
        if search_type == "Tracks" and isrc_filter:
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
        st.subheader(f"Top {len(df)} {search_type} from {search_label}")
        
        # Configure columns for display based on search type
        if search_type == "Tracks":
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
        else:  # Albums
            column_config = {
                "artwork_url": st.column_config.ImageColumn("Artwork", width="small"),
                "name": st.column_config.TextColumn("Album Name", width="medium"),
                "artists": st.column_config.ListColumn(
                    "Artists",
                    help="Artists who created this album",
                    width="medium"
                ),
                "album_type": st.column_config.TextColumn(
                    "Type",
                    help="Type of album (album, single, compilation)",
                    width="small"
                ),
                "total_tracks": st.column_config.NumberColumn(
                    "Tracks",
                    help="Number of tracks in the album",
                    format="%d"
                ),
                "popularity": st.column_config.NumberColumn("Popularity", format="%d"),
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
            
            # Update metric title based on search type
            metric_title = f"Average {search_type[:-1]} Popularity"  # Remove 's' from end
            metric_help = f"The average popularity score across all {search_type_lower} (excluding {search_type_lower} with 0 popularity)"
            
            st.metric(
                metric_title,
                f"{avg_popularity:.1f}",
                help=metric_help
            )
            
            # Create three columns for graphs
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                # Popularity distribution with 10-point bins and colors
                fig_popularity = px.histogram(
                    df,
                    x="popularity",
                    title=f"Distribution of {search_type[:-1]} Popularity",  # Remove 's' from end
                    nbins=10,  # Split by 10
                    color_discrete_sequence=['#1DB954'],  # Spotify green
                    opacity=0.8
                )
                
                # Customize layout
                fig_popularity.update_layout(
                    xaxis_title="Popularity Score",
                    yaxis_title=f"Number of {search_type}",
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
                # Top artists by number of items (tracks or albums)
                top_by_tracks = artist_stats.nlargest(10, 'track_count')
                fig_tracks = px.bar(
                    top_by_tracks,
                    x='artist',
                    y='track_count',
                    title=f"Top 10 Artists by Number of {search_type}",
                    color='track_count',
                    color_continuous_scale='Viridis',
                    labels={
                        'artist': 'Artist',
                        'track_count': f'Number of {search_type}'
                    }
                )
                
                # Customize layout
                fig_tracks.update_layout(
                    xaxis_tickangle=45,
                    showlegend=False,
                    yaxis_title=f"Number of {search_type}"
                )
                
                st.plotly_chart(fig_tracks, use_container_width=True, theme="streamlit")
            
            with col3:
                # Top artists by average popularity (minimum 2 items)
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
                        'track_count': f'Number of {search_type}'
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
                    'track_count': st.column_config.NumberColumn(f"Number of {search_type}")
                },
                hide_index=True,
                use_container_width=True
            ) 
