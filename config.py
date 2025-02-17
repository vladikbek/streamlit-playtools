# Shared configuration constants

# Define common markets with their names
AVAILABLE_MARKETS = {
    'US': 'United States',
    'AR': 'Argentina',
    'AU': 'Australia',
    'BR': 'Brazil',
    'CA': 'Canada',
    'CL': 'Chile',
    'CO': 'Colombia',
    'DE': 'Germany',
    'EC': 'Ecuador',
    'ES': 'Spain',
    'FR': 'France',
    'GB': 'United Kingdom',
    'ID': 'Indonesia',
    'IN': 'India',
    'IT': 'Italy',
    'JP': 'Japan',
    'KR': 'South Korea',
    'KZ': 'Kazakhstan',
    'MY': 'Malaysia',
    'MX': 'Mexico',
    'NL': 'Netherlands',
    'PE': 'Peru',
    'PH': 'Philippines',
    'PL': 'Poland',
    'TH': 'Thailand',
    'TR': 'Turkey',
    'TW': 'Taiwan',
    'UA': 'Ukraine',
    'VN': 'Vietnam'
}

# API configuration
BATCH_SIZE = 20  # Spotify API batch size limit
MAX_WORKERS = 10  # Maximum number of parallel workers
MAX_RESULTS = 1000  # Maximum number of results to return 

# Viral playlists configuration
VIRAL_PLAYLISTS = [
    ('spotify:playlist:37i9dQZEVXbLiRSasKsNU9', 'GLOBAL'),  # Global Viral 50
    ('spotify:playlist:37i9dQZEVXbJajpaXyaKll', 'AR'),      # Argentina Viral 50
    ('spotify:playlist:37i9dQZEVXbK2SUzp56yYx', 'BY'),      # Belarus Viral 50
    ('spotify:playlist:37i9dQZEVXbMOkSwG072hV', 'BR'),      # Brazil Viral 50
    ('spotify:playlist:37i9dQZEVXbJs8e2vk15a8', 'CL'),      # Chile Viral 50
    ('spotify:playlist:37i9dQZEVXbJmRv5TqJW16', 'FR'),      # France Viral 50
    ('spotify:playlist:37i9dQZEVXbNv6cjoMVCyg', 'DE'),      # Germany Viral 50
    ('spotify:playlist:37i9dQZEVXbK4NvPi6Sxit', 'IN'),      # India Viral 50
    ('spotify:playlist:37i9dQZEVXbKpV6RVDTWcZ', 'ID'),      # Indonesia Viral 50
    ('spotify:playlist:37i9dQZEVXbKbvcwe5owJ1', 'IT'),      # Italy Viral 50
    ('spotify:playlist:37i9dQZEVXbINTEnbFeb8d', 'JP'),      # Japan Viral 50
    ('spotify:playlist:37i9dQZEVXbMS8ULOsBprs', 'KZ'),      # Kazakhstan Viral 50
    ('spotify:playlist:37i9dQZEVXbLRmg3qDbY1H', 'MY'),      # Malaysia Viral 50
    ('spotify:playlist:37i9dQZEVXbLuUZrygauiA', 'MX'),      # Mexico Viral 50
    ('spotify:playlist:37i9dQZEVXbMQaPQjt027d', 'NL'),      # Netherlands Viral 50
    ('spotify:playlist:37i9dQZEVXbN7gfhgaomhA', 'PE'),      # Peru Viral 50
    ('spotify:playlist:37i9dQZEVXbJv2Mvelmc3I', 'PH'),      # Philippines Viral 50
    ('spotify:playlist:37i9dQZEVXbNGGDnE9UFTF', 'PL'),      # Poland Viral 50
    ('spotify:playlist:37i9dQZEVXbMIJZxwqzod6', 'TR'),      # Turkey Viral 50
    ('spotify:playlist:37i9dQZEVXbKuaTI1Z1Afx', 'US'),      # USA Viral 50
    ('spotify:playlist:37i9dQZEVXbLwLH0YjrtGb', 'UA'),      # Ukraine Viral 50
    ('spotify:playlist:37i9dQZEVXbL3DLHfQeDmV', 'GB'),      # UK Viral 50
    ('spotify:playlist:37i9dQZEVXbMfVLvbaC3bj', 'ES')       # Spain Viral 50
] 
