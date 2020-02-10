import pandas as pd

from src.database.db_utils import get_collection_from_db

__DB_CONNECTION = get_collection_from_db()


def get_metadata(v_id: str, keys: list) -> dict:
    item = __DB_CONNECTION.find_one({'v_id': v_id})
    if not item:
        raise KeyError(f'V_id: {v_id} not found in database!')
    return dict([(key, item.get(key, 'Failed')) for key in keys])


def get_metaframe(df: pd.DataFrame, keys: list = None) -> pd.DataFrame:
    if keys is None:
        keys = ['v_likes', 'v_dislikes', 'v_duration', 'v_avg_rating']

    df = df.copy()
    df.reindex(columns=[*df.columns.tolist(), *keys])

    for idx, v_id in df.iterrows():
        for key, val in get_metadata(v_id.iloc[0], keys).items():
            df.at[idx, key] = val

    return df
