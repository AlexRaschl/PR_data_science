import pymongo

from src.config import *


def get_collection_from_db():
    conn = pymongo.MongoClient(
        DB_HOST,
        DB_PORT
    )
    db = conn[DB_NAME]
    return db[DB_COLLECTION]
