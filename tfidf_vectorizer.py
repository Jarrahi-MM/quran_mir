# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess_quran_text import merged_quran_vec_df_nrmlz
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np

# %%
corpus = merged_quran_vec_df_nrmlz.fillna(' ').to_numpy().flatten('F').tolist()

# %%
vectorizer = TfidfVectorizer(norm='l2')
doc_term_matrix = vectorizer.fit_transform(corpus)


# %%
def get_most_similars(original_corpus: pd.Series, query: str, K=10) -> pd.DataFrame:
    embedded_query = vectorizer.transform([query])
    cosine_similarities = linear_kernel(doc_term_matrix, embedded_query).flatten()

    quran_len = len(original_corpus)
    similarity_df = pd.DataFrame(data={'original_normalized': cosine_similarities[0:quran_len],
                                       'lemma_normalized': cosine_similarities[quran_len:2 * quran_len],
                                       'root_normalized': cosine_similarities[2 * quran_len:],
                                       },
                                 index=original_corpus.index)
    similarity_series = pd.concat(
        [similarity_df['original_normalized'], similarity_df['root_normalized'], similarity_df['lemma_normalized']],
        axis=1).max(axis=1)
    selected_ayats = similarity_series.sort_values(ascending=False)[:K]

    return pd.DataFrame(data={'آیه': original_corpus[selected_ayats.index],
                              'شباهت': selected_ayats}, index=selected_ayats.index)


# %%
words = vectorizer.get_feature_names()
idf_mat = vectorizer.idf_
median_idf = (np.max(idf_mat) + np.min(idf_mat)) / 2


def get_word_idf(word):
    # should be normalize_and_delete_stopwords
    if word not in words:
        return median_idf
    return idf_mat[words.index(word)]

# %%
