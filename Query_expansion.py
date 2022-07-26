from quran_ir import TfIdfQuranIR
import pandas as pd
import scipy.sparse as sp
from preprocess_quran_text import quran_normalizer, quran_series


# %%
def get_suggested_new_words(suggested_words, original_query):
    new_words = [w for w in suggested_words if w not in original_query]
    return new_words


class QueryExpansion:

    def __init__(self):
        self.tfidf_quran_ir = TfIdfQuranIR()
        self.Quran_aye_num = 6236
        self.params = (1, 0.7, 0.3)
        self.top_similar_ayes_num = 20
        self.dissimilar_ayes_num = 500

    def get_words_in_vocab(self, normalized_query):
        words = normalized_query.split()
        words_in_vocab = [w for w in words if w in self.tfidf_quran_ir.words]
        return words_in_vocab

    def get_avg_emb(self, quran_ayes: pd.DataFrame) -> sp.csr_matrix:
        emb_sum = None
        for i in quran_ayes.index:
            if emb_sum is None:
                emb_sum = self.tfidf_quran_ir.doc_term_matrix[list(quran_series.index).index(i)]
            else:
                emb_sum += self.tfidf_quran_ir.doc_term_matrix[list(quran_series.index).index(i)]
        return emb_sum / len(quran_ayes)

    def expand_query(self, query: str) -> str:
        normalized_query = quran_normalizer(query)
        similarity_df = self.tfidf_quran_ir.get_most_similars(normalized_query, self.Quran_aye_num)
        top_similars = similarity_df[:self.top_similar_ayes_num]
        dissimilars = similarity_df[-self.dissimilar_ayes_num:]
        avg_sim = self.get_avg_emb(top_similars)
        avg_dissim = self.get_avg_emb(dissimilars)
        a, b, c = self.params
        refined_query = a * self.tfidf_quran_ir.vectorizer.transform([normalized_query]) + b * avg_sim - c * avg_dissim
        list1, list2 = zip(*sorted(zip(refined_query.data, refined_query.indices), reverse=True))
        words_in_vocab_len = len(self.get_words_in_vocab(normalized_query))
        top_related = list2[: words_in_vocab_len + min(5, words_in_vocab_len)]
        suggested_words = [self.tfidf_quran_ir.vectorizer.get_feature_names()[e] for e in top_related]
        new_words = get_suggested_new_words(suggested_words, normalized_query)
        expanded_query = normalized_query.split()
        expanded_query.extend(new_words)
        return ' '.join(expanded_query)
