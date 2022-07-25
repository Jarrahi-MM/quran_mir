from quran_ir import FasttextQuranIR
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import normalize
from preprocess_quran_text import quran_series


class QuranRanker:
    def __init__(self, fasttext_quran_ir: FasttextQuranIR):
        self.verse_names = pd.read_csv('data/verse_names.csv')
        self.verse_names.set_index(self.verse_names['ردیف'], inplace=True)
        self.embeddings = fasttext_quran_ir.merged_corpus_embeddings[['original_normalized']].copy()
        self.embeddings['شماره سوره'] = self.embeddings.index.to_series().str.split('##').apply(lambda x: int(x[0]))
        # self.rankings = self.get_ranking(self.embeddings) too long ...

    def get_ranking(self, corpus_embeddings: pd.DataFrame):
        verse_matrix = np.array(corpus_embeddings['original_normalized'].values.tolist())
        P = verse_matrix.dot(verse_matrix.T)
        np.fill_diagonal(P, 0)
        P_norm = normalize(P, norm='l1')
        G = nx.from_numpy_matrix(P_norm, create_using=nx.MultiDiGraph())
        pr = nx.pagerank(G, alpha=0.9)
        h, a = nx.hits(G)
        result = pd.DataFrame(data={'pr': pr, 'a': a, 'h': h})
        result.index = corpus_embeddings.index
        return result

    def get_pivot_aye(self, verse_name):
        verse_number = self.verse_names.loc[self.verse_names['نام سوره'] == verse_name]['ردیف'].index[0]
        r = self.get_ranking(self.embeddings[self.embeddings['شماره سوره'] == verse_number])
        return quran_series[r['a'].sort_values(ascending=False).index[0]]
