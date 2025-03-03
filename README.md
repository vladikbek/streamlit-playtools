# VBR Tools

A Streamlit application that helps users discover and analyze music through various search options including tracks, playlists, and record labels.

## Features

- Search for tracks by various criteria
- Explore playlists and their contents
- Search for record labels and their releases
- Interactive user interface built with Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vladikbek/vbrtools.git
cd vbrtools
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your API credentials:
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.