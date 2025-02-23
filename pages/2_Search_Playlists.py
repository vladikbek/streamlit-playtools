import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import streamlit as st
import plotly.express as px
import concurrent.futures
from typing import List, Dict, Tuple
from collections import Counter
from config import AVAILABLE_MARKETS, BATCH_SIZE, MAX_WORKERS

# Configuration variables
def setup_spotify():
    """Initialize Spotify client with credentials from .env file"""
    load_dotenv()
    auth_manager = SpotifyClientCredentials()
    return spotipy.Spotify(auth_manager=auth_manager)

def search_playlists(sp: spotipy.Spotify, keyword: str, market: str, limit: int = 50) -> list:
    """Search for playlists matching the keyword with pagination support"""
    results = []
    offset = 0
    total_fetched = 0
    
    while total_fetched < limit:
        current_limit = min(50, limit - total_fetched)  # Spotify max is 50 per request
        response = sp.search(q=keyword, type='playlist', limit=current_limit, market=market, offset=offset)
        items = response['playlists']['items']
        
        if not items:  # No more results
            break
            
        results.extend(items)
        total_fetched += len(items)
        offset += len(items)
        
        if len(items) < current_limit:  # Last page
            break
    
    return results

def get_playlist_details(sp: spotipy.Spotify, playlist_id: str) -> dict:
    """Get detailed information about a playlist"""
    return sp.playlist(playlist_id, fields='followers,tracks(total)')

def get_playlist_details_batch(sp: spotipy.Spotify, playlists: List[Dict], batch_size: int = BATCH_SIZE) -> List[Dict]:
    """Get details for a batch of playlists efficiently"""
    processed_playlists = []
    
    # Split playlists into batches
    playlist_batches = [playlists[i:i + batch_size] for i in range(0, len(playlists), batch_size)]
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for batch in playlist_batches:
            future = executor.submit(process_playlist_batch, sp, batch)
            futures.append(future)
            
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            batch_results = future.result()
            if batch_results:
                processed_playlists.extend(batch_results)
    
    return processed_playlists

def process_playlist_batch(sp: spotipy.Spotify, batch: List[Dict]) -> List[Dict]:
    """Process a single batch of playlists in parallel"""
    processed_batch = []
    
    def fetch_playlist_details(playlist_with_index: Tuple[int, Dict]) -> Tuple[int, Dict]:
        """Fetch details for a single playlist while preserving order"""
        index, playlist = playlist_with_index
        try:
            details = sp.playlist(playlist['id'], fields='followers,tracks(total)')
            playlist['followers'] = details['followers']['total']
            playlist['total_tracks'] = details['tracks']['total']
            return index, playlist
        except Exception as e:
            return index, None

    # Create list of (index, playlist) tuples to preserve order
    indexed_playlists = list(enumerate(batch))
    
    # Process playlists in parallel within the batch
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch), MAX_WORKERS)) as executor:
        futures = [executor.submit(fetch_playlist_details, item) for item in indexed_playlists]
        
        # Collect results and maintain original order
        results = []
        for future in concurrent.futures.as_completed(futures):
            index, playlist = future.result()
            if playlist:
                results.append((index, playlist))
    
    # Sort by original index and extract just the playlists
    results.sort(key=lambda x: x[0])
    processed_batch = [playlist for _, playlist in results]
    
    return processed_batch

def search_playlists_parallel(args: Tuple[spotipy.Spotify, str, str, int, bool, int]) -> Tuple[int, List[Dict]]:
    """Search for playlists with keyword and market in parallel"""
    sp, keyword, market, limit, hide_spotify, order_index = args
    try:
        playlists = search_playlists(sp, keyword, market, limit=limit)
        # Add source information and filter out Spotify's playlists if requested
        filtered_playlists = []
        
        # Pre-filter Spotify playlists before getting details
        for position, playlist in enumerate(playlists, 1):
            if hide_spotify and (
                playlist['owner']['display_name'].lower() == 'spotify' or 
                playlist['owner']['id'].lower() == 'spotify'
            ):
                continue
                
            playlist['source_keyword'] = keyword
            playlist['source_market'] = market
            playlist['search_order'] = order_index
            playlist['market_position'] = position
            filtered_playlists.append(playlist)
        
        if not filtered_playlists:
            return order_index, []
            
        # Get details for filtered playlists in batches
        processed_playlists = get_playlist_details_batch(sp, filtered_playlists, BATCH_SIZE)
        return order_index, processed_playlists
    except Exception as e:
        st.warning(f"Search failed for keyword '{keyword}' in market {market}: {str(e)}")
        return order_index, []

def process_playlist_details(args: Tuple[spotipy.Spotify, Dict]) -> Dict:
    """Process playlist details in parallel"""
    sp, playlist = args
    try:
        details = get_playlist_details(sp, playlist['id'])
        playlist['followers'] = details['followers']['total']
        playlist['total_tracks'] = details['tracks']['total']
        return playlist
    except Exception as e:
        st.warning(f"Failed to get details for playlist {playlist['name']}: {str(e)}")
        return None

st.set_page_config(page_title="Search Playlists - Top Songs Finder", page_icon="📑", layout="wide")
st.title("📑 Search Playlists")
st.write("Find popular playlists across Spotify!")

# Initialize Spotify client
sp = setup_spotify()

# Sidebar for configuration
st.sidebar.title("Settings")

# Market selection
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

# Add filtering options
st.sidebar.subheader("Filtering")

# Add option to hide Spotify's playlists
hide_spotify_playlists = st.sidebar.checkbox(
    "Hide Spotify's playlists",
    value=False,
    help="Hide playlists owned by Spotify"
)

results_limit = st.sidebar.slider(
    "Number of results",
    min_value=50,
    max_value=1000,
    value=50,
    step=50,
    help="Maximum number of playlists to fetch per keyword and market (uses pagination)"
)

# Create input field for keyword
search_col1, search_col2 = st.columns([4, 1])  # Ratio 4:1 for input:button

with search_col1:
    keyword = st.text_input("Enter keywords to search for playlists (separate by comma):", "", label_visibility="collapsed")

with search_col2:
    search_button = st.button("Search", type="primary", use_container_width=True)

if search_button and keyword:
    # Split keywords and clean them
    keywords = [k.strip() for k in keyword.split(',') if k.strip()]
    
    if not keywords:
        st.error("Please enter at least one keyword.")
        st.stop()

    all_playlists = []
    total_searches = len(keywords) * len(selected_markets)
    
    with st.spinner(f"Searching for playlists with {len(keywords)} keywords across {len(selected_markets)} markets..."):
        # Create all combinations of keyword and market with order index
        search_args = [
            (sp, keyword, market, results_limit, hide_spotify_playlists, i)
            for i, (keyword, market) in enumerate([(k, m) for k in keywords for m in selected_markets])
        ]
        
        # Process searches in parallel
        ordered_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total_searches)) as executor:
            futures = [executor.submit(search_playlists_parallel, args) for args in search_args]
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                order_index, playlists = future.result()
                ordered_results.append((order_index, playlists))
        
        # Sort results by original order and flatten
        ordered_results.sort(key=lambda x: x[0])
        processed_playlists = [p for _, playlists in ordered_results for p in playlists]

        # Count playlists per owner and track market positions
        owner_counts = {}
        playlist_market_positions = {}
        
        for playlist in processed_playlists:
            owner_id = playlist['owner']['id']
            playlist_id = playlist['id']
            market = playlist['source_market']
            position = playlist['market_position']  # Use the original position
            owner_counts[owner_id] = owner_counts.get(owner_id, 0) + 1
            
            if playlist_id not in playlist_market_positions:
                playlist_market_positions[playlist_id] = {
                    'markets': set(),
                    'positions': [],
                    'playlist': playlist
                }
            
            playlist_market_positions[playlist_id]['markets'].add(market)
            playlist_market_positions[playlist_id]['positions'].append(position)

        # Filter playlists that appear in all selected markets
        cross_market_playlists = []
        for playlist_id, data in playlist_market_positions.items():
            if len(data['markets']) == len(selected_markets):
                # Calculate average position across markets using market_position
                positions = [p['market_position'] for p in processed_playlists 
                           if p['id'] == playlist_id]
                avg_position = sum(positions) / len(positions)
                cross_market_playlists.append((data['playlist'], avg_position))

        # Sort by average position
        cross_market_playlists.sort(key=lambda x: x[1])
        
        # Extract just the playlists, now sorted by average position
        unique_playlists = []
        seen_ids = set()
        
        for playlist, _ in cross_market_playlists:
            playlist_id = playlist['id']
            
            if playlist_id not in seen_ids:
                seen_ids.add(playlist_id)
                unique_playlists.append(playlist)

        # Display search summary with cross-market emphasis
        if len(selected_markets) > 1:
            st.info(f"Found {len(unique_playlists)} playlists that appear in all {len(selected_markets)} selected markets.")
        else:
            st.info(f"Found {len(unique_playlists)} playlists matching your criteria.")

    if not unique_playlists:
        if len(selected_markets) > 1:
            st.error("No playlists found that appear in all selected markets. Try different keywords or fewer markets.")
        else:
            st.error("No playlists found. Try using different keywords.")
        st.stop()

    # Convert to pandas DataFrame
    df = pd.DataFrame([{
        'cover_url': p['images'][0]['url'] if p.get('images') and len(p['images']) > 0 else '',
        'Name': p['name'],
        'Owner': p['owner']['display_name'],
        'Tracks': p['total_tracks'],
        'Followers': p['followers'],
        'Markets': ', '.join(sorted(playlist_market_positions[p['id']]['markets'])),
        'Avg Position': sum(pl['market_position'] for pl in processed_playlists if pl['id'] == p['id']) / len(playlist_market_positions[p['id']]['markets']),
        'Keyword': p['source_keyword'],
        'Description': p.get('description', ''),
        'URL': f"spotify:playlist:{p['id']}",
        'Stats': f"https://www.isitagoodplaylist.com/playlist/{p['id']}",
        'Check': f"/Check_Playlist?playlist={p['id']}"  # Add new Check URL
    } for p in unique_playlists])

    # Display results
    if len(selected_markets) > 1:
        st.subheader(f"Top {len(df)} Cross-Market Playlists")
    else:
        st.subheader(f"Top {len(df)} Playlists")

    # Reset index to start from 1
    df.index = range(1, len(df) + 1)

    # Configure columns
    column_config = {
        "cover_url": st.column_config.ImageColumn(
            "Cover",
            help="Playlist cover image",
            width="small"
        ),
        "URL": st.column_config.LinkColumn(
            "Open",
            display_text="🔗 Open",
            help="Click to open in Spotify desktop/mobile app"
        ),
        "Stats": st.column_config.LinkColumn(
            "Stats",
            display_text="📊 Stats",
            help="View detailed playlist statistics"
        ),
        "Check": st.column_config.LinkColumn(  # Add new Check column
            "Check",
            display_text="🔍 Check",
            help="Analyze playlist tracks"
        ),
        "Followers": st.column_config.NumberColumn(
            "Followers",
            help="Number of followers",
            format="%d"
        ),
        "Tracks": st.column_config.NumberColumn(
            "Tracks",
            help="Number of tracks",
            format="%d"
        ),
        "Avg Position": st.column_config.NumberColumn(
            "Position",
            help="Average position across all markets",
            format="%.1f",
            width="small"
        ),
        "Markets": st.column_config.TextColumn(
            "Found In",
            help="Markets where this playlist appears",
            width="small"
        ),
        "Description": st.column_config.TextColumn(
            "Description",
            width="medium"
        )
    }

    st.dataframe(
        df,
        use_container_width=True,
        column_config=column_config
    )
    
    # Create an expander for analytics
    with st.expander("View Analytics", expanded=False):
        # Total followers
        total_followers = df['Followers'].sum()
        st.metric("Total Followers", f"{total_followers:,}")
        
        # Keyword presence analysis
        def categorize_keyword_presence(row, keyword):
            title = row['Name'].lower()
            description = row['Description'].lower() if pd.notna(row['Description']) else ''
            keyword = keyword.lower()
            
            if keyword in title:
                return 'Keyword in Title'
            elif keyword in description:
                return 'Keyword only in Description'
            else:
                return 'No Keyword Match'
        
        # Get the first keyword (assuming it's the main search term)
        main_keyword = keywords[0] if keywords else ''
        
        # Categorize playlists
        df['keyword_presence'] = df.apply(lambda x: categorize_keyword_presence(x, main_keyword), axis=1)
        keyword_presence_counts = df['keyword_presence'].value_counts()
        
        # Create pie chart for keyword presence
        fig_keyword = px.pie(
            values=keyword_presence_counts.values,
            names=keyword_presence_counts.index,
            title=f"Keyword '{main_keyword}' Presence Analysis"
        )
        fig_keyword.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_keyword, use_container_width=True)
        
        # Top playlist owners
        owner_counts = df['Owner'].value_counts().head(10)
        fig_owners = px.bar(
            x=owner_counts.index,
            y=owner_counts.values,
            title="Top 10 Playlist Creators",
            labels={'x': 'Creator', 'y': 'Number of Playlists'}
        )
        st.plotly_chart(fig_owners, use_container_width=True)
        
        # Enhanced keyword analysis from playlist titles
        def extract_phrases(text):
            """Extract meaningful phrases from text"""
            # Basic stopwords that shouldn't be at the start/end of phrases
            stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            # Clean and normalize text
            text = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in text)
            text = ' '.join(text.split())  # Normalize spaces
            
            words = text.split()
            phrases = []
            
            # Single words (keeping 2+ letter words)
            phrases.extend([w for w in words if len(w) >= 2 and w not in stopwords])
            
            # Look for consecutive words that appear together frequently
            for i in range(len(words) - 1):
                if words[i] not in stopwords:  # Don't start phrase with stopword
                    phrase = words[i]
                    for j in range(i + 1, min(i + 4, len(words))):  # Look up to 3 words ahead
                        if words[j] not in stopwords or j == len(words) - 1:  # Allow stopwords in middle
                            phrase = f"{phrase} {words[j]}"
                            phrases.append(phrase)
            
            return phrases
        
        # Collect phrases from all titles
        all_phrases = []
        for title in df['Name']:
            all_phrases.extend(extract_phrases(title))
        
        # Count and filter phrases
        phrase_counts = Counter(all_phrases)
        
        # Filter meaningful phrases (appear more than once or are multi-word)
        meaningful_phrases = {
            phrase: count for phrase, count in phrase_counts.items()
            if count > 1 or ' ' in phrase  # Keep if multiple occurrences or multi-word
        }
        
        # Get top phrases
        top_phrases = sorted(meaningful_phrases.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Create visualization
        if top_phrases:
            fig_phrases = px.bar(
                x=[phrase for phrase, _ in top_phrases],
                y=[count for _, count in top_phrases],
                title="Most Common Phrases in Playlist Titles",
                labels={'x': 'Phrase', 'y': 'Frequency'}
            )
            fig_phrases.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_phrases, use_container_width=True) 