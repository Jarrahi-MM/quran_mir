from quran_ir import TfIdfQuranIR
import pandas as pd
import scipy.sparse as sp
from preprocess_quran_text import quran_normalizer, quran_series


# %%
class QueryExpansion:

    def __init__(self):
        self.tfidf_quran_ir = TfIdfQuranIR()
        self.Quran_aye_num = 6236
        self.params = (1, 0.7, 0.3)
        self.top_similar_ayes_num = 20
        self.dissimilar_ayes_num = 500

    def get_words_not_in_vocab(self, normalized_query):
        words = normalized_query.split()
        words_not_in_vocab = [w for w in words if w not in self.tfidf_quran_ir.words]
        return words_not_in_vocab

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
        words_not_in_vocab = self.get_words_not_in_vocab(normalized_query)
        words_in_vocab_len = len(normalized_query.split()) - len(words_not_in_vocab)
        top_related = list2[: words_in_vocab_len + min(5, words_in_vocab_len)]
        expanded_query = words_not_in_vocab
        expanded_query.extend([self.tfidf_quran_ir.vectorizer.get_feature_names()[e] for e in top_related
                               # if self.tfidf_quran_ir.vectorizer.get_feature_names()[e] not in stopwords
                               ])
        return ' '.join(expanded_query)



