import numpy as np

def convert_df_to_ndarray(df_data, input_features=['summary_embeddings',
        'category_embeddings',
        'item_type',
        #'agency_id',
        'has_categories']):
    input_feature_list=[]
    for input_feature in input_features:
        input_feature_list.append(np.vstack(df_data[input_feature]))
        X = np.hstack(
            input_feature_list
        )
    return X.astype(np.float64)
