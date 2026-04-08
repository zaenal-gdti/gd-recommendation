import sqlalchemy as sa
import pandas as pd
import numpy as np
import glob
import os
import pandas_gbq
from google.oauth2 import service_account
import gspread 
import gc
from dotenv import load_dotenv
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle
load_dotenv('.env')
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Tuple
from google.cloud import storage
import sys



credentials = service_account.Credentials.from_service_account_file(os.environ['BQ_CRED'])

# Create artifacts directory
if not os.path.exists('artifacts'):
    os.makedirs('artifacts', exist_ok=True)

## Get Data
# Process data in chunks to manage memory
df = pandas_gbq.read_gbq( """
with product_raw as (
  with cte as (
    select id, code as product_code, prescription as need_prescription, deleted,  row_number() over(partition by id order by updated_at desc) rn  from prod_l1_service_inventory.product
  
  )
  select * from cte
  where rn = 1 and deleted is null
), product as (
    with prod_asg as (
    select *, row_number() over(partition by id order by updated_at desc) rn from `prod_l1_service_inventory.product_category_assignment`
    ), prod_exc as (
    select distinct product_id from prod_asg
        where rn = 1 and category_id in (34, 38, 39)
    )
    
    select a.* from product_raw a
    left join prod_exc b on a.id = b.product_id
    where b.product_id is null

), sku as (
  with cte as (
    select id sku_id, product_id, deleted, row_number() over(partition by id order by updated_at desc) rn  from prod_l1_service_inventory.sku
  )
  select * from cte
  where rn = 1 and deleted is null
), prod_sku  as (
  select a.sku_id, a.product_id, b.product_code, need_prescription from  sku a
  left join product b on a.product_id = b.id
)

select a.order_id,created_at,gmv, p.product_id,product_code, sku_name,user_id, profile_id, a.consultation_id, age, gender, icd diagnosis, reddots_flag, 
recommendation_flag, need_prescription
from `l2_reporting.commerce_sku_detail` a
left join (select  distinct consultation_id,     DATE_DIFF(created_at, safe_cast(dob as date), YEAR) - 
    (CASE 
        WHEN FORMAT_DATE('%m%d', created_at) < FORMAT_DATE('%m%d', safe_cast(dob as date)) THEN 1 
        ELSE 0 
    END) AS age , gender, icd from `l2_reporting.consolidate_consult`) b on a.consultation_id = b.consultation_id
left join (
  select order_id, user_id from `l2_reporting.consolidate_commerce`
) c on a.order_id = c.order_id
left join prod_sku p on p.sku_id = cast(a.sku_id as integer)
    where  a.created_at > '2024-02-07' and  p.product_code is not null --DATE_DIFF(current_date, a.created_at, YEAR) < 1
""", credentials = credentials)

# Check if df_all is properly loaded
if df is None or len(df) == 0:
    raise ValueError("Failed to load data from BigQuery or data is empty")

# Free up memory early
gc.collect()

# Check if df is properly created
# if df is None:
#     raise ValueError("DataFrame df could not be created from df")

df['product_id'] = df['product_id'].astype(str)
bins = [0, 1, 5, 12, 19, 39, np.inf]
labels = [
    "1",
    "5",
    "12",
    "19",
    "39",
    "40+",
]

df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True, include_lowest=True).astype(str)


## Healthmall Order history ##
df_hm = df[(df['recommendation_flag'] == False) & (df['reddots_flag'] == False)].copy()

df_hm["created_at"] = pd.to_datetime(df_hm["created_at"])
# Convert timestamp to datetime if it's a Unix timestamp
if df_hm['created_at'].dtype == 'int64':
    df_hm['order_date'] = pd.to_datetime(df_hm['created_at'], unit='s')
else:
    df_hm['order_date'] = df_hm['created_at']

# Create category mappings efficiently
user_categories = df_hm["user_id"].astype("category")
product_categories = df_hm["product_id"].astype("category")

user_codes = user_categories.cat.codes
product_codes = product_categories.cat.codes
user_map = dict(enumerate(user_categories.cat.categories))
product_map = dict(enumerate(product_categories.cat.categories))
user_id_to_int = {v: k for k, v in user_map.items()}

# Create sparse matrix efficiently
interaction_matrix = csr_matrix(
    (np.ones(len(df_hm)), (user_codes, product_codes)),
    shape=(len(user_map), len(product_map))
)

# Store to files - manage memory efficiently
with open('artifacts/user_id_to_int.pkl', 'wb') as f:
    pickle.dump(user_id_to_int, f)

with open('artifacts/product_map.pkl', 'wb') as f:
    pickle.dump(product_map, f)

with open('artifacts/interaction_matrix.pkl', 'wb') as f:
    pickle.dump(interaction_matrix, f)

with open('artifacts/df_hm.pkl', 'wb') as f:
    df_hmx = df_hm[['user_id','product_id', 'sku_name', 'created_at']].copy()
    pickle.dump(df_hmx, f)

# Explicitly delete large objects to free memory
del df_hm, user_categories, product_categories, user_codes, product_codes, user_map, product_map, user_id_to_int, interaction_matrix
gc.collect()
    

## Prescription based ##

# Memory-efficient processing for prescription-based recommendations
# Process data in chunks to prevent memory overflow

df_tmp = df[(df['recommendation_flag'] == True) & (df['reddots_flag'] == False)].copy()

# Process diagnosis codes efficiently
df_tmp['code_diagnosis'] = df_tmp['diagnosis'].astype(str).str.strip().str.replace(' ', '').str.split(",")

# Explode diagnosis codes with memory management
df_exploded = df_tmp.explode("code_diagnosis").reset_index(drop=True)
df_exploded['code_diagnosis'] = df_exploded['code_diagnosis'].astype(str).str.split(".").str[0]
df_exploded['age_gender'] = (
    df_exploded['age_group'].astype(str) + '_' +
    df_exploded['gender'].astype(str)
)

# Process each age/gender group separately to manage memory
cmb_mat = {}

# Get unique age/gender combinations, filtering out null values
valid_age_genders = df_exploded[
    (df_exploded['age_gender'].notna()) & 
    (~df_exploded['age_gender'].str.contains('nan', na=False))
]['age_gender'].unique()

for i in valid_age_genders:
    # Process one age/gender group at a time
    subset = df_exploded[df_exploded['age_gender'] == i]
    
    df_counts = (
        subset
        .drop('diagnosis', axis=1)
        .rename(columns={'code_diagnosis': 'diagnosis'})
        .groupby(["diagnosis", "product_id"])
        .size()
        .reset_index(name="count")
    )
    
    mat = df_counts.pivot_table(
        index="diagnosis",
        columns="product_id",
        values="count",
        aggfunc="sum",
        fill_value=0
    )

    # Compute similarity matrix for this group
    sim_matrix = cosine_similarity(mat.T)
    product_sim = pd.DataFrame(sim_matrix, index=mat.columns, columns=mat.columns)
    cmb_mat[i] = [mat, product_sim]
    
    # Delete temporary variables to free memory
    del subset, df_counts, mat, sim_matrix, product_sim
    gc.collect()

# Clean up temporary dataframes
del df_tmp, df_exploded
gc.collect()

# Get allowed products with memory-efficient processing
allowed = df[(df['reddots_flag'] == False) & (df['recommendation_flag'] == False) & (df['need_prescription'] == False) ].copy()
allowed_products = pd.to_numeric(allowed['product_id'], errors='coerce').dropna().unique().tolist()

# Clean up
del allowed
gc.collect()

# Store to files
with open('artifacts/cmb_mat.pkl', 'wb') as f:
    pickle.dump(cmb_mat, f)

with open('artifacts/allowed_products.pkl', 'wb') as f:
    pickle.dump(allowed_products, f)


## Top N popular product ##
from datetime import datetime, timedelta

def top_hm_product(df_hm, top_n=20):
    cutoff_date = datetime.now() - timedelta(days=90)
    # 2. Filter to last 3 months
    # If created_at is Unix timestamp (like your sample: 1743491933)
    df_recent = df_hm[df_hm['order_date'] >= cutoff_date]
    
    # 3. Count orders per product
    grouped = df_recent.groupby(['product_id', 'sku_name']).size().reset_index(name='order_count')
    sorted_products = grouped.sort_values('order_count', ascending=False)
    top_products = sorted_products.head(top_n).copy()
    
    top_products['score'] = np.nan
    top_products['reason'] = 'Top Product'
    result = top_products.drop('order_count', axis=1)
    
    # Clean up intermediate variables
    del grouped, sorted_products
    return result

top_rec = top_hm_product(pd.read_pickle('artifacts/df_hm.pkl'), top_n=2000)  # Use the saved df_hm to save memory

with open('artifacts/top_rec.pkl', 'wb') as f:
    pickle.dump(top_rec, f)

# Load products data efficiently
products = pandas_gbq.read_gbq("""
with cte as (
select id , code, name, is_active , row_number() over(partition by id order by updated_at) rn from `prod_l1_service_inventory.product`
)

select id as product_id, code as product_code, name as product_name from cte
where rn = 1 and is_active = True
""", credentials=credentials)

with open('artifacts/product_list.pkl', 'wb') as f:
    pickle.dump(products, f)

# Clean up large variables
del df, top_rec, products
gc.collect()

# Upload to cloud storage - fix undefined BUCKET_NAME
BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'default-bucket-name')  # Use environment variable or default

client = storage.Client(credentials=credentials)
bucket = client.bucket(BUCKET_NAME)

for i in glob.glob('artifacts/*'):  # Changed from artifacts_prod to artifacts
    j = os.path.basename(i)
    blob = bucket.blob(f"prod_artifact_rec_engine/{j}")
    blob.upload_from_filename(i)