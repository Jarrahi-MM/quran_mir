# %%
import fasttext
import pandas as pd
import numpy as np
from preprocess_quran_text import quran_normalizer, merged_quran_vec_df_nrmlz, quran_series
from tools import get_most_similars, create_data_set
from quran_ir import TfIdfQuranIR

# %% create date-set
# create_data_set(out_dir='./fasttext_model/data_set.txt', merged_df=merged_quran_vec_df_nrmlz, expansion_count=5,
#                 lemma_rate=0.1, root_rate=0.4)
# %%
# ./fastText/fasttext skipgram -input ./fasttext_model/data_set.txt -output ./fasttext_model/model -ws 5 -dim 100 -minn 3 -maxn 10 -epoch 1000 -thread 15
EMBEDDING_LEN = 100
model = fasttext.load_model('fasttext_model/model.bin')
tfidf_quran_ir = TfIdfQuranIR()


def sent_to_vec(sent: str):
    if pd.isna(sent):
        return np.zeros(EMBEDDING_LEN)
    words = sent.split()
    if len(words) == 0:
        return np.zeros(EMBEDDING_LEN)
    vec = np.average(a=[model.get_word_vector(word) for word in words],
                     weights=[tfidf_quran_ir.get_word_idf(word) for word in words],
                     axis=0)
    return vec / np.linalg.norm(vec)


# %%
merged_corpus_embeddings = merged_quran_vec_df_nrmlz.applymap(sent_to_vec)
