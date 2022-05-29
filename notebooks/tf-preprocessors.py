
def get_categorical_feature_specs(df, col):
    n_uniq = df[col].nunique()
    feature_spec = {
            'n_uniq': n_uniq,
            'num_hash_bins': min(n_uniq, MAX_HASH_BIN_SIZE),
            'emb_size': min(min(n_uniq, MAX_HASH_BIN_SIZE) // 2, MAX_CATEGORICAL_EMBEDDING_SIZE),
    }
    return feature_spec

def define_hasher(df, col, num_hash_bins):
    hasher = Hashing(num_bins=num_hash_bins, salt=1337, name=f'{col}_hasher')
    return hasher

def define_normalizer(df, col):
    normalizer = Normalization(axis=None, name=f'{col}_normalizer')
    normalizer.adapt(df[col].values)
    return normalizer

def preprocessors(data_dict):

    processors = {}
    feature_specs = {}
    
    for k, df in data_dict.items():
        columns_ = [col for col in df.columns if col not in DEP_FEATURES]
        for col in columns_:
            if col in CAT_FEATURES + FLAT_CAT_FEATURES:
                feature_spec = get_categorical_feature_specs(df, col)
                feature_specs[col] = feature_spec
                processors[col] = define_hasher(df, col, feature_spec['num_hash_bins'])
            if col in CONT_FEATURES:
                processors[col] = define_normalizer(df, col)
        for col in DEP_FEATURES:
            feature_specs[col] = feature_specs['article_id']
            processors[col] = processors['article_id']
    
    return processors, feature_specs
processors, feature_specs = preprocessors(data)