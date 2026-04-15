# Playlist Tools

Playlist Tools is a Streamlit app for searching and analyzing Spotify playlists, tracks, and label catalogs.

## Features

- Search playlists across markets
- Inspect full playlist track data
- Search label catalogs as tracks or releases
- Analyze playlist SEO positions across Spotify markets

## Installation

1. Clone the repository and enter the project directory:

```bash
git clone https://github.com/vladikbek/streamlit-playtools.git
cd streamlit-playtools
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies:

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Spotify credentials:

```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

## Usage

Run the app inside the virtual environment:

```bash
streamlit run Main_Page.py
```

Streamlit will print the local URL in the terminal.
