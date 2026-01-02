import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import streamlit as st
import concurrent.futures
from typing import List, Dict, Tuple
from app.config import AVAILABLE_MARKETS, BATCH_SIZE, MAX_WORKERS

# Configuration variables
def get_query_param(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value

def parse_markets_param(value: str | None) -> List[str]:
    if not value:
        return []
    markets = []
    for item in value.split(','):
        code = item.strip().upper()
        if code and code in AVAILABLE_MARKETS:
            markets.append(code)
    return markets

def parse_bool_param(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

def parse_int_param(value: str | None, default: int, min_value: int, max_value: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, number))

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

def search_playlists_parallel(args: Tuple[spotipy.Spotify, str, str, int, int]) -> Tuple[int, List[Dict]]:
    """Search for playlists with keyword and market in parallel"""
    sp, keyword, market, limit, order_index = args
    try:
        playlists = search_playlists(sp, keyword, market, limit=limit)
        # Add source information
        filtered_playlists = []
        
        # Pre-filter playlists before getting details
        for position, playlist in enumerate(playlists, 1):
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

def is_official_playlist(playlist: Dict) -> bool:
    """Check if a playlist is owned by Spotify (official)."""
    owner = playlist.get('owner') or {}
    owner_id = (owner.get('id') or '').lower()
    owner_name = (owner.get('display_name') or '').lower()
    return owner_id == 'spotify' or owner_name == 'spotify'

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

st.title(":material/search: Search Playlists", anchor=False)
st.caption("Find playlists across countries with a few keywords, see what shows up everywhere.")

# Initialize Spotify client
sp = setup_spotify()

param_keyword = get_query_param("q") or ""
param_markets = parse_markets_param(get_query_param("countries"))
param_limit = parse_int_param(get_query_param("limit"), 50, 50, 1000)
param_hide_official = parse_bool_param(get_query_param("hide_official"))

# Create input field for keyword
with st.container(border=True):
    with st.form("playlist_search_form", border=False):
        search_col1, search_col2, search_col3 = st.columns([8, 2, 3])

        with search_col1:
            keyword = st.text_input(
                "Enter keywords to search for playlists (separate by comma):",
                param_keyword,
                label_visibility="collapsed",
                placeholder="Paste keywords, split with commas (e.g. phonk, hard techno)",
                icon=":material/link:"
            )

        with search_col2:
            results_limit = st.number_input(
                "Number of results",
                min_value=50,
                max_value=1000,
                value=param_limit,
                step=50,
                help="Maximum number of playlists to fetch per keyword and market (uses pagination)",
                icon=":material/post:",
                label_visibility="collapsed"
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
                "Countries",
                options=list(AVAILABLE_MARKETS.keys()),
                default=param_markets or ['US'],
                format_func=lambda x: f"{x} - {AVAILABLE_MARKETS[x]}",
                help="Select markets to search playlists in",
                label_visibility="collapsed",
                placeholder="Select countries to search playlists in"
            )

            hide_official_playlists = st.checkbox(
                "Hide official playlists",
                value=param_hide_official,
                help="Hide playlists owned by Spotify"
            )


# If no markets selected, default to US
if not selected_markets:
    selected_markets = ['US']

current_search_key = {
    "q": keyword.strip(),
    "countries": ",".join(selected_markets),
    "limit": str(int(results_limit))
}
last_search_key = st.session_state.get("last_search_key")
auto_search = bool(param_keyword)
should_run_search = bool(keyword) and (
    search_button or (auto_search and current_search_key != last_search_key)
)

if should_run_search:
    # Split keywords and clean them
    keywords = [k.strip() for k in keyword.split(',') if k.strip()]

    if not keywords:
        st.error("Please enter at least one keyword.")
        st.stop()

    total_searches = len(keywords) * len(selected_markets)

    with st.spinner(f"Searching for playlists with {len(keywords)} keywords across {len(selected_markets)} markets..."):
        # Create all combinations of keyword and market with order index
        search_args = [
            (sp, keyword, market, results_limit, i)
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

    st.session_state["search_results"] = {
        "processed_playlists": processed_playlists,
        "selected_markets": selected_markets,
        "keywords": keywords
    }
    st.session_state["last_search_key"] = current_search_key

    if search_button:
        st.query_params.from_dict({
            "q": keyword.strip(),
            "countries": ",".join(selected_markets),
            "limit": str(int(results_limit)),
            "hide_official": "1" if hide_official_playlists else "0"
        })

search_results = st.session_state.get("search_results")
if search_results:
    processed_playlists = search_results["processed_playlists"]
    selected_markets = search_results["selected_markets"]
    keywords = search_results["keywords"]

    # Track market positions
    playlist_market_positions = {}

    for playlist in processed_playlists:
        playlist_id = playlist['id']
        market = playlist['source_market']
        position = playlist['market_position']  # Use the original position

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

    if not unique_playlists:
        if len(selected_markets) > 1:
            st.error("No playlists found that appear in all selected markets. Try different keywords or fewer markets.")
        else:
            st.error("No playlists found. Try using different keywords.")
        st.stop()

    display_playlists = unique_playlists
    if hide_official_playlists:
        display_playlists = [p for p in unique_playlists if not is_official_playlist(p)]

    # Display search summary with cross-market emphasis
    if len(selected_markets) > 1:
        st.info(f"Found {len(display_playlists)} playlists that appear in all {len(selected_markets)} selected markets.")

    if not display_playlists:
        st.warning("No playlists left after filtering. Turn off 'Hide official playlists' to see them.")
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
        'Check': f"/check_playlist?playlist={p['id']}"  # Add new Check URL
    } for p in display_playlists])

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
            "Link",
            display_text=":material/open_in_new:",
            help="Click to open in Spotify desktop/mobile app",
            width="small"
        ),
        "Name": st.column_config.TextColumn(
            "Name",
            width="medium"
        ),
        "Description": st.column_config.TextColumn(
            "Description",
            width="medium"
        ),
        "Owner": st.column_config.TextColumn(
            "Owner",
            width="medium"
        ),
        "Stats": st.column_config.LinkColumn(
            "Stats",
            display_text=":material/query_stats:",
            help="View detailed playlist statistics",
            width="small"
        ),
        "Check": st.column_config.LinkColumn(  # Add new Check column
            "Check",
            display_text=":material/playlist_add_check:",
            help="Analyze playlist tracks",
            width="small"
        ),
        "Followers": st.column_config.NumberColumn(
            "Followers",
            help="Number of followers",
            format="%d",
            width="small"
        ),
        "Tracks": st.column_config.NumberColumn(
            "Tracks",
            help="Number of tracks",
            format="%d",
            width="small"
        ),
        "Keyword": st.column_config.TextColumn(
            "Keyword",
            width="small"
        ),
        "Avg Position": st.column_config.NumberColumn(
            "Position",
            help="Average position across all markets",
            format="%.1f",
            width="small"
        ),
        "Markets": st.column_config.TextColumn(
            "Countries",
            help="Markets where this playlist appears",
            width="small"
        )
    }

    st.dataframe(
        df,
        width="stretch",
        height=500,
        column_order=[
            "URL",
            "cover_url",
            "Name",
            "Description",
            "Owner",
            "Followers",
            "Tracks",
            "Keyword",
            "Markets",
            "Avg Position",
            "Stats",
            "Check"
        ],
        column_config=column_config
    )
