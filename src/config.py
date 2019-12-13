API_KEY = 'AIzaSyCj9-VpsoNOKui6MpJUPKPWTa8tnj1Mv98'

LFM_PATH = r'data/LFM-1b.txt'

YT_WATCH_STUB = r'https://www.youtube.com/watch?v='
YT_SEARCH_STUB = r'https://www.youtube.com/results?search_query='

CACHE_PATH = 'cache'
EMPTY_PATH = 'NONE'
FAIL_PATH = 'FAILED'

# N_LINKS_SEARCHED = 5

# Database configs
DB_HOST = 'localhost'
DB_PORT = 27017
DB_NAME = 'musicvideos'
DB_COLLECTION = 'yt_final'

# Crawl Configs
N_CRAWLS = 5000

# Download Configs
DL_DELAY = 10
DL_PATH = "H:\\Datasets\\YouTubeFinal\\"
JSON_INFO_EXTENSION = '.info.json'

# Preprocessing configs
RES_RSCLD = (480, 360)
FFMPEG_PATH = "H:\Anwendungen\\ffmpeg\\bin\\ffmpeg.exe"

N_SAMPLES = 30
SAMPLE_OFFSET = 10
SIMILARITY_THRES = 0.95

INDEXED_TTS_PATH = CACHE_PATH + '\\tts_42'
STORED_PRED_PATH = CACHE_PATH + '\\stored'

# Testing constants
TRAIN_SIZE = 1894
TEST_SIZE = 211
