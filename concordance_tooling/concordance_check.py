from src.ml_resources import (
    read_dataset_from_file
)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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

def rm_text_please(original_text, ignore_ls):
    """
    remove a sentence (from ignore_ls) from a string (original_text)
    orginal_text:
        original string
    ignore_ls:
        a list
    output: new_text
    """
    # Removes the brackets and everything inside them
    rm_curling_brackets = re.sub(r'\{.*?\}', '', original_text)
    result = re.sub(r'\[.*?\]', '', rm_curling_brackets)
    # Regex Approach: For better performance with many replacements, join the list into a regular expression pattern.
    pattern = "|".join(ignore_ls)
    new_text = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return new_text


import spacy
import re
nlp = spacy.load('en_core_web_sm')
ignore_file = '../concordance_tooling/ignore_sentences.txt'
with open(ignore_file, 'r') as file:
        ignore_ls = [line.strip() for line in file]

agency='uk.iser.ukhls'
usoc_embeddings = read_dataset_from_file(f'./projects/am1_project/data/filtered_embeddings_{agency}/filtered_embeddings_{agency}_1.pickle')
agency='uk.whitehall2'
whitehall2_embeddings = read_dataset_from_file(f'./projects/am1_project/data/filtered_embeddings_{agency}/filtered_embeddings_{agency}_1.pickle')



usoc_embeddings_array=convert_df_to_ndarray(
    usoc_embeddings['X_train'], input_features=['TextLabel_embeddings']
    )
usoc_embeddings['X_train']['Topic']=usoc_embeddings['y_train']
topic='10601'
usoc_embeddings['X_train'][usoc_embeddings['X_train']['Topic']==topic]
embeddings_10703=usoc_embeddings['X_train'][usoc_embeddings['X_train']['Topic']==topic]


whitehall2_embeddings['X_train']['Topic']=whitehall2_embeddings['y_train']
whitehall2_embeddings['X_train'][whitehall2_embeddings['X_train']['Topic']==topic]
whitehall2_embeddings_10703=whitehall2_embeddings['X_train'][whitehall2_embeddings['X_train']['Topic']==topic]

whitehall2_embeddings_10703['TextLabel'] = whitehall2_embeddings_10703['TextLabel'].apply(lambda x: rm_text_please(str(x), ignore_ls))
whitehall2_embeddings_10703['nouns_verbs_lemmas'] = whitehall2_embeddings_10703['TextLabel'].apply(lambda x: ([token.lemma_ for token in nlp(x) if token.pos_ in ['NOUN', 'VERB'] and not token.is_stop and len(token) > 1]))
vectorizer = TfidfVectorizer(max_features=10000)
label_tdf=vectorizer.fit_transform(
    [str(x) for x in whitehall2_embeddings_10703['nouns_verbs_lemmas'].values.tolist()])
whitehall2_embeddings_10703['tdf_vector']=label_tdf.toarray().tolist()




search=3776
df1 = pd.concat([embeddings_10703.loc[[search]], whitehall2_embeddings_10703], ignore_index=True)
vectorizer = TfidfVectorizer(max_features=10000)
label_tdf=vectorizer.fit_transform(
    [str(x) for x in df1['nouns_verbs_lemmas'].values.tolist()])
df1['tdf_vector']=label_tdf.toarray().tolist()

all_embeddings_array=convert_df_to_ndarray(df1,
    input_features=['TextLabel_embeddings', 'ItemCategories_embeddings', 'tdf_vector']
    )
embedding_similarity = cosine_similarity(all_embeddings_array)

all_embeddings_array=convert_df_to_ndarray(df1,
    input_features=['TextLabel_embeddings', 'ItemCategories_embeddings']
    )
embedding_similarity = cosine_similarity(all_embeddings_array)


indices = np.argpartition(embedding_similarity[0], -20)[-20:]
values = embedding_similarity[0][indices]
print(indices)
print(values)
df1['TextLabel'].iloc[indices].values



vector1 = np.concatenate(
    df1.iloc[0][['TextLabel_embeddings', 'ItemCategories_embeddings']].values
)
vector2 = np.concatenate(
    df1.iloc[64][['TextLabel_embeddings', 'ItemCategories_embeddings']].values
)