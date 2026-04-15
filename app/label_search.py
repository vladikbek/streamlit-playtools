import pandas as pd
import spotipy

from app.config import MAX_RESULTS


def search_items_by_label(
    sp: spotipy.Spotify,
    label: str,
    search_type: str,
    limit: int = 50,
    year: int | None = None,
) -> list:
    """Search Spotify tracks or albums by record label."""
    items = []
    offset = 0

    query = f'label:"{label}"'
    if year:
        query += f" year:{year}"

    response_key = "tracks" if search_type == "track" else "albums"

    while True:
        results = sp.search(q=query, type=search_type, limit=limit, offset=offset)
        batch = results[response_key]["items"]
        if not batch:
            break

        items.extend(batch)
        offset += limit

        if offset >= MAX_RESULTS:
            break

    return items


def search_tracks_by_label(
    sp: spotipy.Spotify,
    label: str,
    limit: int = 50,
    year: int | None = None,
) -> list:
    """Search for tracks by record label."""
    return search_items_by_label(sp, label, search_type="track", limit=limit, year=year)


def search_releases_by_label(
    sp: spotipy.Spotify,
    label: str,
    limit: int = 50,
    year: int | None = None,
) -> list:
    """Search for releases by record label."""
    return search_items_by_label(sp, label, search_type="album", limit=limit, year=year)


def process_tracks_batch(args):
    """Process a batch of tracks."""
    sp, tracks_batch = args
    processed_tracks = []

    for track in tracks_batch:
        try:
            artists = []
            for artist in track.get("artists", []):
                if artist and isinstance(artist, dict) and artist.get("name"):
                    artists.append(artist["name"])

            if not artists:
                continue

            images = track.get("album", {}).get("images", [])
            artwork_url = images[-1]["url"] if images else ""
            isrc = track.get("external_ids", {}).get("isrc", "N/A")

            processed_tracks.append(
                {
                    "artwork_url": artwork_url,
                    "name": track.get("name", "Unknown Track"),
                    "artists": artists,
                    "isrc": isrc,
                    "popularity": track.get("popularity", 0),
                    "release_date": track.get("album", {}).get("release_date", ""),
                    "url": track.get("uri", ""),
                }
            )
        except Exception:
            continue

    return processed_tracks


def process_releases_batch(args):
    """Process a batch of releases."""
    sp, releases_batch = args
    processed_releases = []
    release_ids = [release.get("id") for release in releases_batch if release.get("id")]
    album_details = {}

    for index in range(0, len(release_ids), 20):
        response = sp.albums(release_ids[index:index + 20])
        for album in response.get("albums", []):
            if album and album.get("id"):
                album_details[album["id"]] = album

    for release in releases_batch:
        try:
            album = album_details.get(release.get("id"))
            if not album:
                continue

            artists = []
            for artist in album.get("artists", []):
                if artist and isinstance(artist, dict) and artist.get("name"):
                    artists.append(artist["name"])

            if not artists:
                continue

            images = album.get("images") or []
            artwork_url = images[-1]["url"] if images else ""
            external_ids = album.get("external_ids") or {}

            processed_releases.append(
                {
                    "id": album.get("id", ""),
                    "artwork_url": artwork_url,
                    "name": album.get("name", "Unknown Release"),
                    "artists": artists,
                    "label": album.get("label", "Unknown Label"),
                    "album_type": (album.get("album_type") or "unknown").replace("_", " ").title(),
                    "total_tracks": album.get("total_tracks", 0),
                    "upc": external_ids.get("upc", "N/A"),
                    "popularity": album.get("popularity", 0),
                    "release_date": album.get("release_date", ""),
                    "url": album.get("uri", ""),
                }
            )
        except Exception:
            continue

    return processed_releases


def parse_release_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    try:
        date_str = str(date_str)
        if len(date_str) == 4:
            return pd.to_datetime(f"{date_str}-01-01")
        if len(date_str) == 7:
            return pd.to_datetime(f"{date_str}-01")
        return pd.to_datetime(date_str)
    except Exception:
        return pd.NaT
