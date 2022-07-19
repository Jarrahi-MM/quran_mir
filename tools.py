import numpy as np
import pandas as pd
import qalsadi.lemmatizer
from nltk.stem.isri import ISRIStemmer
import tqdm

stemmer = ISRIStemmer()
lemmer = qalsadi.lemmatizer.Lemmatizer()  # This is a weak Lemmatizer.


def create_data_set(out_dir: str, merged_df: pd.DataFrame, lemma_rate=0, root_rate=0, expansion_count=0):
    with open(out_dir, 'w') as fp:
        for col in merged_df.columns:
            for aye in tqdm.tqdm(merged_df[col].dropna().tolist()):
                fp.write(aye)
                fp.write('\n')
        probabilities = [1 - lemma_rate - root_rate, lemma_rate, root_rate]
        functions = [lambda x: x, lemmer.lemmatize, stemmer.stem]
        for _ in range(expansion_count):  # can be enhanced
            for aye in tqdm.tqdm(merged_df['original_normalized']):
                fp.write(' '.join([np.random.choice(a=functions, p=probabilities)(token) for token in aye.split()]))
                fp.write('\n')


def similarity(query_vec: np.ndarray):
    def temp(sent_vec: np.ndarray):
        return sent_vec @ query_vec

    return temp


def get_most_similars(original_corpus: pd.Series, merged_corpus_embeddings: pd.DataFrame,
                      query_vec: np.ndarray, K=10, check_moghattaeh=False) -> pd.DataFrame:
    '''
    :param check_moghattaeh:
    :param K:
    :param original_corpus: uses this data to produce the result df.
    :param merged_corpus_embeddings: may have many columns. for originals, roots, lemmas, ... . each row is for an aye. embeddings should be normalized.
    :param query_vec:
    :return: ayat and scores df.
    '''
    similarity_df = merged_corpus_embeddings.applymap(similarity(query_vec))

    similarity_series = pd.concat(
        [similarity_df['original_normalized'], similarity_df['root_normalized'], similarity_df['lemma_normalized']],
        axis=1).max(axis=1)
    if not check_moghattaeh:
        selected_ayats = similarity_series.sort_values(ascending=False)[:K]
    else:
        sorted_similarities = similarity_series.sort_values(ascending=False)
        count = 0
        selected_ayats = {}
        for i in sorted_similarities.index:
            if len(original_corpus[i].split()) > 1:
                count += 1
                selected_ayats[i] = sorted_similarities[i]
            if count == K:
                break
        selected_ayats = pd.Series(selected_ayats)
        selected_ayats = selected_ayats.sort_values(ascending=False)

    return pd.DataFrame(data={'آیه': original_corpus[selected_ayats.index],
                              'شباهت': selected_ayats}, index=selected_ayats.index)
