# Playlist Tools

Playlist Tools is a Streamlit application that helps users discover and analyze music through various search options including tracks, playlists, and record labels using Spotify API.

## Features

- Search for tracks by various criteria
- Explore playlists and their contents
- Search for record labels and their releases
- Interactive user interface built with Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vladikbek/playtools.git
cd vbrtools
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your API credentials:
```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

## Usage

Run the Streamlit app:
```bash
streamlit run Main_Page.py
```

The application will open in your default web browser.
