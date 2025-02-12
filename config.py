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