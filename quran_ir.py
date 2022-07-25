from abc import abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess_quran_text import merged_quran_vec_df_nrmlz, quran_series, quran_normalizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np


class QuranIR:

    def __init__(self):
        self.algorithm = '?'

    @abstractmethod
    def get_most_similars(self, original_corpus: pd.Series, query: str, K=10, check_moghattaeh=False) -> pd.DataFrame:
        raise NotImplementedError("Please Implement this method")

    def process_queries(self, query_path='./queries.txt', result_path='ir_responses/') -> pd.DataFrame:
        with open(query_path) as f:
            queries = f.readlines()
            queries = [q.strip() for q in queries]

        results = []
        i = 1
        for query in queries:
            results.append({'Query': 'q{} = "{}"'.format(i, query)})
            results.extend(self.get_most_similars(quran_series, quran_normalizer(query), 10).to_dict('records'))
            i += 1
        results = pd.DataFrame(results)
        results['شباهت'] = results['شباهت'].round(3, )
        results.to_csv(f'{result_path}{self.algorithm}.csv', index=False)
        return results


class TfIdfQuranIR(QuranIR):

    def __init__(self):
        super().__init__()
        self.algorithm = 'TfIdf'
        corpus = merged_quran_vec_df_nrmlz.fillna(' ').to_numpy().flatten('F').tolist()
        self.vectorizer = TfidfVectorizer(norm='l2')
        self.doc_term_matrix = self.vectorizer.fit_transform(corpus)

        self.words = self.vectorizer.get_feature_names()
        self.idf_mat = self.vectorizer.idf_
        self.median_idf = (np.max(self.idf_mat) + np.min(self.idf_mat)) / 2

    def get_most_similars(self, original_corpus: pd.Series, query: str, K=10, check_moghattaeh=False) -> pd.DataFrame:
        embedded_query = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(self.doc_term_matrix, embedded_query).flatten()

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

    def get_word_idf(self, word):
        # should be normalize_and_delete_stopwords
        if word not in self.words:
            return self.median_idf
        return self.idf_mat[self.words.index(word)]
