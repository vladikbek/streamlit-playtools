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
from urllib.parse import quote

def get_query_param(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value

def split_input(value: str) -> List[str]:
    if not value:
        return []
    items = []
    for part in value.replace("\n", ",").split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items

def parse_search_depth(value: str | None, default: int = 100) -> int:
    try:
        depth = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(50, min(1000, depth))

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

st.title(":material/analytics: Playlist SEO", anchor=False)
st.caption("Track how your playlists rank for each keyword across all markets.")

# Initialize Spotify client
sp = setup_spotify()

playlists_param = get_query_param("playlists") or ""
keywords_param = get_query_param("keywords") or ""

with st.container(border=True):
    with st.form("playlist_seo_form", border=False):
        col1, col2 = st.columns([1, 1])

        with col1:
            playlist_url = st.text_area(
                "Enter Spotify playlist URLs/URIs:",
                value=playlists_param,
                placeholder="One per line or comma-separated",
                label_visibility="collapsed",
                height=120
            )

        with col2:
            keywords = st.text_area(
                "Enter keywords to analyze:",
                value=keywords_param,
                placeholder="One per line or comma-separated",
                label_visibility="collapsed",
                height=120
            )

        action_col1, action_col2 = st.columns([8, 3])

        with action_col1:
            search_depth_input = st.text_input(
                "Search depth per keyword",
                value="100",
                help="How deep to search in results (50-1000)",
                label_visibility="collapsed",
                placeholder="Search depth per keyword (50-1000)"
            )

        with action_col2:
            search_button = st.form_submit_button(
                "Analyze SEO Position",
                type="primary",
                icon=":material/search:",
                width="stretch"
            )

last_search_key = st.session_state.get("seo_search_key")
auto_search = bool(playlists_param or keywords_param)
current_search_key = {
    "playlists": playlist_url.strip(),
    "keywords": keywords.strip(),
    "depth": search_depth_input.strip()
}
should_run = bool(playlist_url and keywords) and (
    search_button or (auto_search and current_search_key != last_search_key)
)

if should_run:
    results_limit = parse_search_depth(search_depth_input, default=100)

    # Extract playlist IDs
    playlist_urls = split_input(playlist_url)
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
    
    # Split and clean keywords
    keyword_list = split_input(keywords)
    
    if not keyword_list:
        st.error("Please enter at least one keyword.")
        st.stop()
    
    # Always search all markets
    markets_to_search = list(AVAILABLE_MARKETS.keys())
    
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
    
    st.session_state["seo_search_key"] = current_search_key
    st.session_state["seo_results"] = {
        "results": results,
        "playlists": playlist_url,
        "keywords": keywords,
        "depth": results_limit,
        "playlist_infos": playlist_infos
    }

    if search_button:
        st.query_params.from_dict({
            "playlists": playlist_url.strip(),
            "keywords": keywords.strip()
        })

seo_results = st.session_state.get("seo_results")
if seo_results:
    results = seo_results["results"]

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure positions are integers where found
    df['position'] = df['position'].apply(lambda x: int(x) if pd.notnull(x) else x)
    
    # Prepare detailed results DataFrame
    detailed_df = df[['playlist_name', 'playlist_id', 'keyword', 'market', 'position']].copy()
    
    # Create a numeric position column for sorting
    detailed_df['sort_position'] = detailed_df['position'].apply(lambda x: float('inf') if pd.isna(x) else x)
    
    # Convert position to string, handling NaN values
    detailed_df['position'] = detailed_df['position'].apply(lambda x: str(int(x)) if pd.notnull(x) else 'Not found')
    # Add full market names
    detailed_df['country'] = detailed_df['market'].apply(lambda x: [AVAILABLE_MARKETS[x]])
    detailed_df['playlist_open'] = detailed_df['playlist_id'].apply(
        lambda x: f"https://open.spotify.com/playlist/{x}"
    )
    detailed_df['playlist_link'] = detailed_df.apply(
        lambda row: f"https://open.spotify.com/playlist/{row['playlist_id']}#name={row['playlist_name']}",
        axis=1
    )
    detailed_df['search_link'] = detailed_df.apply(
        lambda row: f"/search_playlists?q={quote(row['keyword'])}&countries={row['market']}",
        axis=1
    )
    
    # Sort by position (found positions first, then "Not found")
    detailed_df = detailed_df.sort_values(['sort_position', 'playlist_name'])
    
    # Drop the sorting column and set final column names
    detailed_df = detailed_df.drop('sort_position', axis=1)
    detailed_df = detailed_df.rename(columns={
        "playlist_link": "Playlist",
        "playlist_open": "Link",
        "search_link": "Search",
        "keyword": "Keyword",
        "country": "Country",
        "position": "Position"
    })
    display_df = detailed_df[["Link", "Playlist", "Keyword", "Country", "Position", "Search"]].copy()

    st.subheader(f"Found {len(display_df)} results", anchor=False)
    st.caption("All markets searched for every keyword and playlist.")

    primary_color = st.get_option("theme.primaryColor")
    results_tab, summary_tab = st.tabs(["Results", "Summary"])

    with results_tab:
        st.dataframe(
            display_df,
            hide_index=True,
            width="stretch",
            height=500,
            column_config={
                "Link": st.column_config.LinkColumn(
                    "Link",
                    display_text=":material/open_in_new:",
                    width="small"
                ),
                "Playlist": st.column_config.LinkColumn(
                    "Playlist",
                    display_text=r"#name=(.*)$",
                    width="large"
                ),
                "Keyword": st.column_config.TextColumn(
                    "Keyword",
                    width="medium"
                ),
                "Country": st.column_config.ListColumn(
                    "Country",
                    width="medium"
                ),
                "Position": st.column_config.TextColumn(
                    "Position",
                    width="small"
                ),
                "Search": st.column_config.LinkColumn(
                    "Search",
                    display_text=":material/search:",
                    width="small"
                )
            }
        )

    with summary_tab:
        found_mask = display_df["Position"] != "Not found"
        found_count = int(found_mask.sum())
        total_count = len(display_df)
        st.metric("Found", f"{found_count}")
        st.metric("Not Found", f"{total_count - found_count}")

        if found_count > 0:
            summary = (
                display_df[found_mask]
                .assign(Position=lambda x: x["Position"].astype(int))
                .groupby("Keyword", as_index=False)
                .agg(avg_position=("Position", "mean"))
                .sort_values("avg_position")
            )
            fig_positions = px.bar(
                summary,
                x="Keyword",
                y="avg_position",
                title="Average Position by Keyword",
                labels={"avg_position": "Average Position"}
            )
            if primary_color:
                fig_positions.update_traces(marker_color=primary_color)
            fig_positions.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_positions, width="stretch")
