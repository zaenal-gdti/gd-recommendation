
import pickle
import logging
import asyncio
from typing import Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Query
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from google.cloud import storage
import gspread
import math

from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. State Container
# ---------------------------------------------------------
@dataclass
class RecEngineState:
    '''Holds a snapshot of all loaded ML models and datasets.'''
    interaction_matrix: Any
    df_hm: pd.DataFrame
    top_rec: pd.DataFrame
    cmb_mat: dict
    user_id_to_int: dict
    allowed_list: list
    sku_map: dict
    product_map: dict
    product_list: pd.DataFrame

# ---------------------------------------------------------
# 2. Recommender Engine Logic
# ---------------------------------------------------------
class HealthRecommender:
    def __init__(self, data_dir: str = '/tmp/rec_engine_data'):
        '''Initializes the recommender manager using /tmp for Cloud Run compatibility.'''
        self.data_dir = Path(data_dir)
        self.state: Optional[RecEngineState] = None
        self.promoted_products: pd.DataFrame = pd.DataFrame()  # Pre-initialize to prevent AttributeError

    def _load_pickle(self, filename: str) -> Any:
        bucket_name = 'gdt-ml-eng'
        file_path = self.data_dir / filename

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f'Failed to remove old file {file_path}: {e}')

        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(filename)

            if blob.exists():
                logger.info(f'Downloading freshest {filename} from GCS...')
                blob.download_to_filename(file_path)
            else:
                logger.error(f'FATAL: GCS Blob {filename} does not exist.')
                return None

        except Exception as e:
            logger.error(f'GCS download failed for {filename}: {e}.')
            return None

        try:
            if not file_path.exists():
                return None
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f'Error unpickling {filename}: {e}')
            return None

    def _load_all_sync(self) -> RecEngineState:
        logger.info('Starting concurrent artifact load...')

        paths = {
            'interaction_matrix': 'artifact_rec_engine/interaction_matrix.pkl',
            'df_hm': 'artifact_rec_engine/df_hm.pkl',
            'top_rec': 'artifact_rec_engine/top_rec.pkl',
            'cmb_mat': 'artifact_rec_engine/cmb_mat.pkl',
            'user_id_to_int': 'artifact_rec_engine/user_id_to_int.pkl',
            'allowed_list': 'artifact_rec_engine/allowed_products.pkl',
            'product_map': 'artifact_rec_engine/product_map.pkl',
            'product_list': 'artifact_rec_engine/product_list.pkl'
        }

        results = {}
        with ThreadPoolExecutor(max_workers=len(paths)) as executor:
            future_to_key = {
                executor.submit(self._load_pickle, path): key 
                for key, path in paths.items()
            }
            for future in future_to_key:
                results[future_to_key[future]] = future.result()

        df_hm = results['df_hm']
        sku_map = {}
        if df_hm is not None and not df_hm.empty:
            sku_map = df_hm.set_index('product_id')['sku_name'].to_dict()

        logger.info('All artifacts loaded into memory.')
        return RecEngineState(
            interaction_matrix=results['interaction_matrix'],
            df_hm=df_hm,
            top_rec=results['top_rec'],
            cmb_mat=results['cmb_mat'],
            user_id_to_int=results['user_id_to_int'] or {},
            allowed_list=results['allowed_list'] or [],
            sku_map=sku_map,
            product_map=results['product_map'] or {},
            product_list=results['product_list']
        )

    async def reload(self):
        try:
            new_state = await asyncio.to_thread(self._load_all_sync)
            self.state = new_state
            logger.info('Successfully swapped to new recommender state.')
        except Exception as e:
            logger.error(f'Failed to reload recommender artifacts: {e}')

    def get_collaborative_recs(self, target_user_id: int, top_k_users: int = 10) -> pd.DataFrame:
        state = self.state
        if not state or state.interaction_matrix is None or not state.user_id_to_int:
            return pd.DataFrame()

        target_idx = state.user_id_to_int.get(target_user_id)
        if target_idx is None:
            return pd.DataFrame()

        user_vector = state.interaction_matrix[target_idx]
        sim_scores = cosine_similarity(user_vector, state.interaction_matrix).flatten()
        similar_user_indices = sim_scores.argsort()[-(top_k_users+1):-1][::-1]

        cf_scores = {}
        for idx in similar_user_indices:
            score = sim_scores[idx]
            if score <= 0: continue
            neighbor_items = state.interaction_matrix[idx].indices
            for item_int in neighbor_items:
                cf_scores[item_int] = cf_scores.get(item_int, 0) + score

        final_recs = []
        for item_int, score in cf_scores.items():
            p_code = state.product_map.get(item_int)
            if p_code:
                sku = state.sku_map.get(p_code, 'Unknown Product')
                final_recs.append({
                    'product_id': p_code,
                    'sku_name': sku,
                    'reason': 'Discovery'
                })

        return pd.DataFrame(final_recs)

    def get_buy_again(self, user_id: int, limit: int = 5) -> pd.DataFrame:
        state = self.state
        if not state or state.df_hm is None:
            return pd.DataFrame()

        df_hist = state.df_hm[state.df_hm['user_id'] == user_id].copy()
        if df_hist.empty:
            return pd.DataFrame()

        df_hist = (
            df_hist
            .sort_values('created_at', ascending=False)
            .drop_duplicates('product_id', keep='first')
            .iloc[:limit]
        )

        df_hist['reason'] = 'Buy Again'
        return df_hist[['product_id', 'sku_name', 'reason']]

    def get_prescription_recs(
        self, current_diagnosis: str, anchor_product: str, age: int, gender: str, top_n: int = 5
    ) -> pd.DataFrame:
        state = self.state
        if not state or state.cmb_mat is None:
            return pd.DataFrame()

        current_diagnosis = current_diagnosis.strip().replace(' ', '').split('.')[0]

        bins = [0, 1, 5, 12, 19, 39, np.inf]
        labels = ['1', '5', '12', '19', '39', '40+']
        age_group = pd.cut([age], bins=bins, labels=labels)[0]
        group_key = f'{age_group}_{gender}'

        matrix_data = state.cmb_mat.get(group_key)
        if not matrix_data:
            return pd.DataFrame()

        mat, product_sim = matrix_data
        if current_diagnosis not in mat.index or anchor_product not in product_sim.columns:
            return pd.DataFrame()

        sim_series = product_sim[anchor_product].sort_values(ascending=False)
        sim_series = sim_series.drop(anchor_product, errors='ignore')

        diag_row = mat.loc[current_diagnosis]
        products_for_diag = diag_row[diag_row > 0].index

        valid_candidates = (
            sim_series.index
            .intersection(set(state.allowed_list))
            .intersection(products_for_diag)
        )

        candidates = sim_series.loc[valid_candidates].head(top_n)

        if candidates.shape[0] == 0:
            return pd.DataFrame()

        df_rec = pd.DataFrame({'product_id': candidates.index})
        df_rec['reason'] = 'Prescription based order'

        # FIX: Use .copy() to avoid modifying the in-memory singleton DataFrame
        prod_list = state.product_list.copy()
        prod_list['product_id'] = pd.to_numeric(prod_list['product_id'], errors='coerce')
        df_rec['product_id'] = pd.to_numeric(df_rec['product_id'], errors='coerce')

        df_rec = df_rec.merge(prod_list, how='left', on='product_id').rename(columns={'product_name': 'sku_name'})
        return df_rec[['product_id', 'sku_name', 'reason']].drop_duplicates('product_id')

    def get_top_products(self, top_n: int = 200) -> pd.DataFrame:
        state = self.state
        if not state or state.top_rec is None:
            return pd.DataFrame()

        top_products = state.top_rec.copy()
        top_products['reason'] = 'Top Product'
        return top_products[['product_id', 'sku_name', 'reason']].head(top_n)

    def _load_promoted_sync(self) -> pd.DataFrame:
        sheet_id = '1EQpmd81YGk8lCQwceaM9dHOa2G6AHJ3_XyLRmYKnREA'

        try:
            logger.info('Fetching promoted products from Google Sheets...')
            gc = gspread.service_account('cred_gsheet.json')
            sht1 = gc.open_by_key(sheet_id)
            worksheet = sht1.worksheet('Sheet1')

            # FIX: get_all_records handles headers and empty rows safely
            records = worksheet.get_all_values(range_name = 'A1:B10000')
            if not records:
                return pd.DataFrame()
                
            df = pd.DataFrame(records)
            cols = df.iloc[0,]
            df = df.drop(0)
            df.columns = cols
            
            # FIX: Guard clause in case APScheduler fires before ML models load
            state = self.state
            if not state or state.product_list is None:
                logger.warning('Engine state not ready. Skipping promoted products merge.')
                return pd.DataFrame()

            # Normalize column names for merging

            df = df.merge(state.product_list, how='left', on='product_code')
            df = df.rename(columns={'product_name': 'sku_name'})
            
            df = df[['product_id', 'sku_name']]
            df['score'] = 999
            df['reason'] = 'Promoted Product'
            logger.info(f'Loaded {len(df)} promoted products.')
            return df

        except Exception as e:
            logger.error(f'Failed to fetch Google Sheet: {e}')
            return pd.DataFrame()

    async def reload_promoted(self):
        new_promo_df = await asyncio.to_thread(self._load_promoted_sync)
        if not new_promo_df.empty:
            self.promoted_products = new_promo_df
            logger.info('Successfully updated promoted products state.')

    def get_sb(self) -> pd.DataFrame:
        return self.promoted_products

# ---------------------------------------------------------
# 3. FastAPI Lifespan and Endpoints Integration
# ---------------------------------------------------------
rec_engine = HealthRecommender()

@asynccontextmanager
async def lifespan(app: FastAPI):
    FastAPICache.init(InMemoryBackend(), prefix='fastapi-cache')
    logger.info('FastAPI cache initialized.')

    await rec_engine.reload()
    await rec_engine.reload_promoted()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(rec_engine.reload, 'interval', hours=12, id='reload_rec_artifacts', replace_existing=True)
    scheduler.add_job(rec_engine.reload_promoted, 'interval', minutes=5, id='reload_promoted_products', replace_existing=True)

    scheduler.start()
    logger.info('APScheduler started: ML(12h), Promo(5m).')
    yield
    scheduler.shutdown()
    logger.info('APScheduler shut down.')

app = FastAPI(lifespan=lifespan)

@app.get('/sbp_recommendations')
@cache(expire=300)  # Caches the specific page request for 5 minutes
async def sbp_recommendations(
    page: int = Query(1, ge=1, description='Page number (starts at 1)'),
    limit: int = Query(20, ge=1, le=100, description='Items per page')
):
    df_sbp = rec_engine.get_sb()

    if df_sbp.empty:
        return {
            'data': [],
            'meta': {'total_items': 0, 'page': page, 'limit': limit, 'total_pages': 0}
        }

    total_items = len(df_sbp)
    total_pages = math.ceil(total_items / limit)

    start_idx = (page - 1) * limit
    end_idx = start_idx + limit

    paginated_df = df_sbp.iloc[start_idx:end_idx]

    return {
        'data': paginated_df.to_dict(orient='records'),
        'meta': {
            'total_items': total_items,
            'page': page,
            'limit': limit,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
    }


@app.post('/recommendations')
@cache(expire=3600)
async def get_user_recommendations(
    user_id: int, 
    age: Optional[int] = None, 
    gender: Optional[str] = None,
    diagnosis: Optional[List[str]] = Query(None), 
    cart_products: Optional[List[str]] = Query(None),
    page: int = Query(1, ge=1, description='Page number'),
    limit: int = Query(20, ge=1, le=100, description='Items per page')
):
    if rec_engine.state is None:
        return {'error': 'Engine is warming up, please try again in a few seconds.'}

    buy_gn = rec_engine.get_buy_again(user_id=user_id)
    hm_rec = rec_engine.get_collaborative_recs(target_user_id=user_id)
    top_prod = rec_engine.get_top_products(top_n=100)

    rec_diag_all = pd.DataFrame()
    if age and gender and diagnosis and cart_products:
        rec_diag = []
        for d in diagnosis:
            for c in cart_products:
                z = rec_engine.get_prescription_recs(d, c, age, gender)
                if not z.empty:
                    rec_diag.append(z)
        if rec_diag:
            rec_diag_all = pd.concat(rec_diag, ignore_index=True).drop_duplicates('product_id').dropna()

    # FIX: Removed duplicate variable assignment 
    dfs_to_concat = [df for df in [buy_gn, hm_rec, rec_diag_all, top_prod] if not df.empty]

    if not dfs_to_concat:
        return {
            'data': [],
            'meta': {'total_items': 0, 'page': page, 'limit': limit, 'total_pages': 0}
        }

    out = pd.concat(dfs_to_concat, ignore_index=True)
    out = out.drop_duplicates('product_id', keep='first')

    total_items = len(out)
    total_pages = math.ceil(total_items / limit)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_df = out.iloc[start_idx:end_idx]

    return {
        'data': paginated_df.to_dict(orient='records'),
        'meta': {
            'total_items': total_items,
            'page': page,
            'limit': limit,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
    }


if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=int(8080))
