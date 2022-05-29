# # Strategy:
# 1. Most popular baseline (in the training data)
# 2. Repeat last ordered item baseline (in the training data)
# 3. Experiments
#     1. Users & Products only
#     2. Order features
#     3. Customer features
#     4. Add date parts
#     5. Product features
#         1. Product metadata
#         2. Product Image
#         3. Product Description
# 

import pandas as pd
import numpy as np
import os, gc, pickle, time
from sklearn.decomposition import PCA
from pandas_tfrecords import pd2tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

start_time = time.perf_counter()

pd.options.display.max_columns = 500
seeded_value = 8888
pd.set_option('display.max_colwidth', 50)
np.random.seed(seeded_value)

# suppress scientific notation
pd.options.display.precision = 2
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

REDUCE_MEM_USAGE = False
IS_KAGGLE = False
REDUCE_DIMENSIONALITY = False

MAX_SEQ_LEN = 10
MAX_CATEGORICAL_EMBEDDING_SIZE = 20
MAX_HASH_BIN_SIZE = 20
TEXT_EMB_DIM = 768
IMAGE_EMB_DIM = 2048
NUM_SAMPLES = 20_000
MAX_DATES_LEN = 56


if IS_KAGGLE:
    DATA_DIR = '../input/h-and-m-personalized-fashion-recommendations/'
    IMAGE_DIR = '../input/hm-image-features-w-resnet50/'
    TEXT_DIR = '../input/hm-text-features-w-roberta/'
    RESULTS_DIR = ''
else:
    DATA_DIR = '../data/'
    IMAGE_DIR = '../data/'
    TEXT_DIR = '../data/'
    RESULTS_DIR = '../results/'


csv_list = [os.path.join(DATA_DIR, p) for p in os.listdir(DATA_DIR) if p.endswith('.csv') if p != 'sample_submission.csv']

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# reading all csv
data = {}
for file_path in csv_list:
    file_name = file_path.split('.')[-2].split("/")[-1].strip(" ")
    print(f"Reading {file_name}.csv")
    if REDUCE_MEM_USAGE:
        data[file_name] = reduce_mem_usage(pd.read_csv(file_path))
    else:
        data[file_name] = pd.read_csv(file_path)

for k, v in data.items():
    print(f"******** {k} ********")
    print(v.head(2))

for k, v in data.items():
    print(f"******** {k} ********")
    print(v.shape, v.columns.tolist())

def reduce_dimensionality(features, n_components):
    features_ = np.array(features.values.tolist())
    pca_ = PCA(n_components=n_components)
    features_decomposed = pca_.fit_transform(features_)
    return features_decomposed.tolist()

# ## Date features

def preprocess_dates(data, datecolname):
    data[datecolname] = pd.to_datetime(data[datecolname], format = ("%Y-%m-%d"))
    # print({
    #     "min_": data["date_time"].min(),
    #     "max_": data["date_time"].max(),
    #     "nunique_": data["date_time"].nunique()
    # })
    data['year_dt'] = data[datecolname].dt.year.astype('int16')
    data['month_dt'] = data[datecolname].dt.month.astype('int16')
    data['day_dt'] = data[datecolname].dt.day.astype('int16')
    data['weekofyear_dt'] = data[datecolname].dt.isocalendar().week.astype('int16')
    data['dayofweek_dt'] = data[datecolname].dt.dayofweek.astype('int16') + 1 
    data['dayofyear_dt'] = data[datecolname].dt.dayofyear.astype('int16')
    data['quarter_dt'] = data[datecolname].dt.quarter.astype('int16')
    data['is_month_start_dt'] = data[datecolname].dt.is_month_start.astype('int16')
    data['is_month_end_dt'] = data[datecolname].dt.is_month_end.astype('int16')
    data['is_quarter_start_dt'] = data[datecolname].dt.is_quarter_start.astype('int16')
    data['is_quarter_end_dt'] = data[datecolname].dt.is_quarter_end.astype('int16')
    data['is_year_start_dt'] = data[datecolname].dt.is_year_start.astype('int16')
    data['is_year_end_dt'] = data[datecolname].dt.is_year_end.astype('int16')
    data['is_leap_year_dt'] = data[datecolname].dt.is_leap_year.astype('int16')
    data['daysinmonth_dt'] = data[datecolname].dt.daysinmonth.astype('int16')
    return data

dates_set_array = pd.to_datetime(data['transactions_train']['t_dat'], format = ("%Y-%m-%d")).sort_values().unique()
data['dates'] = preprocess_dates(pd.DataFrame({'t_dat': dates_set_array}), 't_dat')

# ## Image features

with open(os.path.join(IMAGE_DIR, 'image_df.pkl'), 'rb') as f:
    image_df = pickle.load(f)
    image_df['article_id'] = np.where(image_df['article_id'].isna(), "-1", image_df['article_id'])
    image_df['article_id'] = image_df['article_id'].astype(np.int64)
    
    image_df['image_features'] = image_df['image_features'].apply(lambda x: x.tolist())
    image_df = image_df[['article_id', 'image_features']].copy()
    if REDUCE_DIMENSIONALITY:
        image_df['image_features'] = reduce_dimensionality(image_df['image_features'], 10)

# ## Text features

with open(os.path.join(TEXT_DIR, 'text_df.pkl'), 'rb') as f:
    text_df = pickle.load(f)
    text_df['article_id'] = np.where(text_df['article_id'].isna(), "-1", text_df['article_id'])
    text_df['article_id'] = text_df['article_id'].astype(np.int64)
    
    text_df['detail_desc_features'] = text_df['detail_desc_features'].apply(lambda x: x.tolist())
    text_df = text_df[['article_id', 'detail_desc_features']].copy()
    gc.collect()
    if REDUCE_DIMENSIONALITY:
        text_df['detail_desc_features'] = reduce_dimensionality(text_df['detail_desc_features'], 10)

# ## Customer Feature Extraction Pipeline
# 
# FN is if a customer get Fashion News newsletter, Active is if the customer is active for communication, sales channel id, 2 is online and 1 store.
# 
# Grouping postal codes based on sales and number of customers

data['customers'].head()

data['customers'].isna().sum()

data['customers'].info()

data['customers'].nunique()

data['customers'].describe()

data['customers']['age'] = data['customers']['age'].astype('float32')
data['customers']['club_member_status'] = data['customers']['club_member_status'].str.lower()
data['customers']['fashion_news_frequency'] = data['customers']['fashion_news_frequency'].str.lower()

missing_value_impute_dict = {
    'FN': 0.0,
    'Active': 0.0,
    'club_member_status': 'not-applicable',
    'fashion_news_frequency': 'none',
    'age': np.round(data['customers']['age'].mean())
}

for col, impute_value in missing_value_impute_dict.items():
    data['customers'][col] = np.where(data['customers'][col].isna(), impute_value, data['customers'][col])

{col:data['customers'][col].unique() for col in data['customers'] if col not in ['customer_id', 'postal_code']}

# ## Key observations
# 1. The transaction data is not at the correct level and hence will need to be aggregated to `t_dat`, `article_id`, `customer_id`, `sales_channel_id` , `price` level  and `qty` column to be created to adjust for the missing information (28805603 rows vs 31788324 rows)
# 2. `article_id` and `product_code` seem to map n-to-1
# 3. Submission dataset has some customers which are not present in transaction file or customer file
# `data['sample_submission']['customer_id'].nunique(), data['customers']['customer_id'].nunique(), data['transactions_train']['customer_id'].nunique()` --> 1371980, 1371980, 1362281
# 4. Breaking the data into 7 day rolling periods can be a good way generate data (a lot of it)

# ## Transactions Feature Extraction Pipeline

data['transactions_train'].shape

data['transactions_train'].nunique()

data['transactions_train'].info()

data['transactions_train']['t_dat'] = pd.to_datetime(data['transactions_train']['t_dat'], format = ("%Y-%m-%d"))

data['transactions_train']['t_dat'].describe()

# ## Preprocessing

CAT_FEATURES = [
    'article_id', 'product_code', 'product_type_no', 'product_type_name', 'product_group_name', 'graphical_appearance_no', 'graphical_appearance_name',
    'colour_group_code', 'colour_group_name', 'perceived_colour_value_id', 'perceived_colour_value_name', 'perceived_colour_master_id',
    'perceived_colour_master_name', 'department_no', 'department_name', 'index_code', 'index_name', 'index_group_no', 'index_group_name',
    'section_no', 'section_name', 'garment_group_no', 'garment_group_name' # products
]

CAT_FEATURES = CAT_FEATURES + [
    'FN', 'Active', 'club_member_status', 'fashion_news_frequency', 'postal_code' #customers
] + [
    'sales_channel_id',  # transactions
    'month_dt', 'day_dt', 'week_dt', 'weekofyear_dt', 'dayofweek_dt', 'dayofyear_dt', 'quarter_dt',  # dates
]

FLAT_CAT_FEATURES = ['customer_id'] # customers

CONT_FEATURES = [
    'age', # customers
    'price', 'qty', # transactions
    'is_month_start_dt', 'is_month_end_dt', 'is_quarter_start_dt', 'is_quarter_end_dt', # dates
    'is_year_start_dt', 'is_year_end_dt', 'is_leap_year_dt', 'daysinmonth_dt', 'year_dt', # dates
]

TEXT_FEATURES = ['detail_desc_features']

IMAGE_FEATURES = ['image_features']

FEATURES = CAT_FEATURES + FLAT_CAT_FEATURES + CONT_FEATURES + TEXT_FEATURES + IMAGE_FEATURES

DEP_FEATURES = ['y']

def force_categories_to_int64(x):
    if x.dtype in [np.float16, np.float32, np.float64]:
        return x.astype(np.int64)
    return x

def add_unknown_category(x):
    if x.dtype in [np.int16, np.int32, np.int64]:
        res = np.concatenate([x.values, np.array([-1])])
    else:
        res = np.concatenate([x.values, np.array(['unknown'])])
    return res

encoding_dict = {}
for col in data['articles'].columns:
    if col in CAT_FEATURES + FLAT_CAT_FEATURES:
        data['articles'][col] = force_categories_to_int64(data['articles'][col])
        label_enc = LabelEncoder()
        label_enc.fit(add_unknown_category(data['articles'][col]))
        data['articles'][col] = label_enc.transform(data['articles'][col]) + 1
        encoding_dict[col] = label_enc
    if col in CONT_FEATURES:
        standard_enc = StandardScaler()
        data['articles'][[col]] = standard_enc.fit_transform(data['articles'][[col]])
        encoding_dict[col] = standard_enc

gc.collect()

for col in data['dates'].columns:
    if col in CAT_FEATURES + FLAT_CAT_FEATURES:
        data['dates'][col] = force_categories_to_int64(data['dates'][col])
        label_enc = LabelEncoder()
        label_enc.fit(add_unknown_category(data['dates'][col]))
        data['dates'][col] = label_enc.transform(data['dates'][col]) + 1
        encoding_dict[col] = label_enc
    if col in CONT_FEATURES:
        standard_enc = StandardScaler()
        data['dates'][[col]] = standard_enc.fit_transform(data['dates'][[col]])
        encoding_dict[col] = standard_enc

gc.collect()

for col in data['customers'].columns:
    if col in CAT_FEATURES + FLAT_CAT_FEATURES:
        data['customers'][col] = force_categories_to_int64(data['customers'][col])
        label_enc = LabelEncoder()
        label_enc.fit(add_unknown_category(data['customers'][col]))
        data['customers'][col] = label_enc.transform(data['customers'][col]) + 1
        encoding_dict[col] = label_enc
    if col in CONT_FEATURES:
        standard_enc = StandardScaler()
        data['customers'][[col]] = standard_enc.fit_transform(data['customers'][[col]])
        encoding_dict[col] = standard_enc

gc.collect()

for col in data['transactions_train'].columns:
    if col == 'sales_channel_id':
        data['transactions_train'][col] = force_categories_to_int64(data['transactions_train'][col])
        label_enc = LabelEncoder()
        label_enc.fit(add_unknown_category(data['transactions_train'][col]))
        data['transactions_train'][col] = label_enc.transform(data['transactions_train'][col]) + 1
        encoding_dict[col] = label_enc
    if col == 'price':
        standard_enc = StandardScaler()
        data['transactions_train'][[col]] = standard_enc.fit_transform(data['transactions_train'][[col]])
        encoding_dict[col] = standard_enc
    if col in ['customer_id', 'article_id']:
        data['transactions_train'][col] = encoding_dict[col].transform(data['transactions_train'][col]) + 1

gc.collect()

text_df['article_id'] = encoding_dict['article_id'].transform(text_df['article_id'])
image_df['article_id'] = encoding_dict['article_id'].transform(image_df['article_id'])

def truncate_and_add_padding(x: list, max_seq_len: int, padding_value: int=0):
    dtype_ = type(x[0])
    x = x[-max_seq_len:]
    len_ = len(x)
    return np.array([padding_value] * (max_seq_len - len_) + x, dtype=dtype_).tolist()

def get_missing_image_vector(image_df):
    MISSING_PLACEHOLDER = encoding_dict['article_id'].transform([-1])
    impute_vector = image_df['image_features'].loc[image_df['article_id'] == MISSING_PLACEHOLDER[0]].iloc[0]
    return impute_vector

def merge_additional_info(dataset):
    
    results = dataset.merge(data['dates'], on='t_dat', how='left')
    results = results.merge(data['customers'], on='customer_id', how='left')
    results = results.merge(data['articles'][[col for col in data['articles'].columns if col in FEATURES]], on='article_id', how='left')
    missing_image_vector = get_missing_image_vector(image_df)
    results = results.merge(image_df[['article_id', 'image_features']], on='article_id', how='left')
    results['image_features'] = results['image_features'].apply(lambda x: missing_image_vector if x is np.nan or x is None else x)
    results = results.merge(text_df[['article_id', 'detail_desc_features']], on='article_id', how='left')
    return results

def make_list_uniq(list_):
    return list(set(list_))

def slice_and_agg(dates_subset, dates_subset_y, verbose=False):
    data_slice_y = data['transactions_train'].loc[(data['transactions_train']['t_dat'].isin(dates_subset_y))].copy()
    data_slice_y = data_slice_y.groupby('customer_id', as_index=False).agg({
        'article_id': lambda x: x.tolist()
    }).rename(columns={'article_id':'y'})[['customer_id', 'y']]
    gc.collect()
    if verbose: print(data_slice_y['customer_id'].nunique(), "customers found in validation period..")
    
    data_slice = data['transactions_train'].loc[(data['transactions_train']['t_dat'].isin(dates_subset))].copy()
    data_slice = data_slice.loc[(data['transactions_train']['customer_id'].isin(data_slice_y['customer_id'].unique()))].copy()
    data_slice['qty'] = 1
    gc.collect()

    if verbose: print(data_slice['customer_id'].nunique(), "customers found in training period..")
    
    TXN_GROUP_COLS = ['t_dat', 'customer_id', 'sales_channel_id', 'article_id', 'price']

    data_slice = data_slice.groupby(TXN_GROUP_COLS, as_index=False).agg({'qty': 'sum'}).sort_values([
        't_dat', 'customer_id', 'sales_channel_id', 'article_id', 'price'
    ],ascending=[
        True, True, True, True , True
    ])

    data_slice = data_slice.sample(NUM_SAMPLES).copy()
    gc.collect()
    
    data_slice = merge_additional_info(data_slice)
    data_slice = data_slice.groupby('customer_id', as_index=False).agg({
        col: lambda x: x.tolist()
        for col in data_slice.columns
        if col not in ['customer_id', 't_dat', 'date_time']
    }).reset_index(drop=True)
    
    results = data_slice.merge(data_slice_y, how='inner', on='customer_id')
    if verbose: print(results['customer_id'].nunique(), "customers found in the final dataset..")
    gc.collect()
    
    results['image_features'] = results['image_features'].apply(lambda x : np.array(x).mean(axis=0).tolist())
    results['detail_desc_features'] = results['detail_desc_features'].apply(lambda x : np.array(x).mean(axis=0).tolist())
    results['y'] = results['y'].apply(lambda x : make_list_uniq(x))
    gc.collect()
    return results

# ## Setting up the train validation and CV

def remove_indices_above_max_index(x, t=(len(dates_set_array)-1)):
    return [i for i in x if i <= t]

train_validation_indices = [
    (
        list(range(0, i * 28)),
        list(range(i * 28, (i + 1) * 28)),
        list(range((i + 1) * 28, (i + 2) * 28)),
    )
    for i in range((len(dates_set_array) // 28) + 1)
    if i != 0
]

print(len(train_validation_indices))


# for n, (train_i, valid_i, test_i) in enumerate(train_validation_indices):
#     n = str(n)
#     print("starting", n, "out of", len(train_validation_indices))

#     train_i = remove_indices_above_max_index(train_i)
#     valid_i = remove_indices_above_max_index(valid_i)
#     test_i = remove_indices_above_max_index(test_i)

#     t_i = train_i[-MAX_DATES_LEN:]
#     tv_i = train_i + valid_i
#     tv_i = tv_i[-MAX_DATES_LEN:]

#     train_df = slice_and_agg(dates_set_array[t_i], dates_set_array[valid_i])
#     gc.collect()
    
#     valid_df = slice_and_agg(dates_set_array[tv_i], dates_set_array[test_i])
#     gc.collect()
    
#     pd2tf(df=train_df, folder=os.path.join(RESULTS_DIR, f'train-tfrecords-idx-{n}'), compression_level=6, max_mb=50)
#     pd2tf(df=valid_df, folder=os.path.join(RESULTS_DIR, f'valid-tfrecords-idx-{n}'), compression_level=6, max_mb=50)
    
#     del train_df
#     del valid_df
#     gc.collect()

with open(os.path.join(RESULTS_DIR, 'encoding_dict.pkl'), 'wb') as f:
    pickle.dump(encoding_dict, f)

end_time = time.perf_counter()
duration = end_time - start_time

print(f"Done in {duration} seconds...")

#### nohup python preprocess-data.py >> preprocess-data-$(date +"%Y%m%d").log 2>&1 &