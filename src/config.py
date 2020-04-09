import multiprocessing

API_KEY = 'INSERT'

# DATA
LFM_PATH = r'data/LFM-1b.txt'
LABEL_TO_STRING_PATH = r'data/imagenet1000_clsidx_to_labels.txt'

YT_WATCH_STUB = r'https://www.youtube.com/watch?v='
YT_SEARCH_STUB = r'https://www.youtube.com/results?search_query='

CACHE_PATH = 'cache'
EMPTY_PATH = 'NONE'
FAIL_PATH = 'FAILED'
DELETED_PATH = 'DELETED'

# N_LINKS_SEARCHED = 5

# Database configs
DB_HOST = 'localhost'
DB_PORT = 27017
DB_NAME = 'musicvideos'
DB_COLLECTION = 'yt_final'

# Crawl Configs
N_CRAWLS = 35000

# Download Configs
DL_DELAY = 20
DL_PATH = "H:\\Datasets\\YouTubeFinal\\"
JSON_INFO_EXTENSION = '.info.json'

# Preprocessing configs
RES_RSCLD = (480, 360)
FFMPEG_PATH = "H:\Anwendungen\\ffmpeg\\bin\\ffmpeg.exe"
DELETE_FP = True

N_SAMPLES = 30
SAMPLE_OFFSET = 10
SIMILARITY_THRES = 0.95

# CACHE
SPLIT_SEED = 42
INDEXED_TTS_PATH = CACHE_PATH + '\\tts_' + str(SPLIT_SEED)
STORED_PRED_PATH = INDEXED_TTS_PATH + '\\cnn_pred'
STORED_FACE_PATH = INDEXED_TTS_PATH + '\\faces'
STORED_COLOR_PATH = INDEXED_TTS_PATH + '\\colors'

CACHED_LABEL_STRINGS = CACHE_PATH + '\\labels\\labels.pkl'

# Testing constants
TRAIN_SIZE = 9344
TEST_SIZE = 1039

# GridSearchCV
N_JOBS = int(multiprocessing.cpu_count() * (2 / 3))
GRID_SEARCH_LOG_FOLDER = 'logs\\gscv'
SEARCH_METRICS = ('neg_mean_absolute_error',)
FILE_CREATION_MODE = 'w'
N_FOLDS = 5

# Visualization Path
VIS_PATH = 'visualisations\\'

# Face Cascade
CASCADE_PATH = 'data/haarcascade_frontalface_default.xml'

# Baseline kw
BL_DATA_DICT = {'cnn_ds': True, 'cnn_agg': False, 'n_labels': 1, 'ohe_cnn': True}
FULL_DATA_DICT = {'duration_ds': True,
                  'cnn_ds': True,
                  'color_ds': True,
                  'face_ds': True,
                  'cnn_agg': True,
                  'ohe_cnn': False,
                  'ohe_color': True,
                  'n_labels': None
                  }
# Standard Preprocessing Dict
PP_DICT = {'n_components': -1, 'std_scale': True, 'with_mean': True, 'log_tf': True}  # No PCA, Normalize with Mean

# Model Storage
STORED_MODEL_PATH = 'models'
