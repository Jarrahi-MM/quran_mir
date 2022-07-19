# %% import section
import pandas as pd
from camel_tools.utils.normalize import *
from camel_tools.utils.dediac import *
import tqdm
import numpy as np
from parsi_io.modules.quranic_extractions import QuranicExtraction
import codecs

# %% read data
quran_df = pd.read_csv("data/Quran.txt", sep='\t', header=None)
quran_series = pd.Series(data=quran_df[1].tolist(), index=quran_df[0])
quran_root_df = pd.read_csv("data/Quran_root.txt", sep='\t', header=None)
quran_lemma_df = pd.read_csv("data/Quran_lemma.txt", sep='\t', header=None)
verse_complete_dict = pd.Series(quran_df[1].tolist(), index=quran_df[0]).to_dict()
verse_root_dict = pd.Series(quran_root_df[1].tolist(), index=quran_root_df[0]).to_dict()
verse_lemma_dict = pd.Series(quran_lemma_df[1].tolist(), index=quran_lemma_df[0]).to_dict()


# %% normalization

def choose_normalizer(normalizer_type):
    if normalizer_type == "parsi-io":
        extractor = QuranicExtraction()

        def parsi_io_normalizer(sent):
            return extractor.run(sent)

        return parsi_io_normalizer


quran_normalizer = choose_normalizer("parsi-io")
verse_complete_dict_nrmlz = {k: quran_normalizer(v) for k, v in tqdm.tqdm(verse_complete_dict.items())}
verse_root_dict_nrmlz = {k: quran_normalizer(v) for k, v in tqdm.tqdm(verse_root_dict.items())}
verse_lemma_dict_nrmlz = {k: quran_normalizer(v) for k, v in tqdm.tqdm(verse_lemma_dict.items())}

# %%
merged_quran_df = pd.concat({'index': quran_df[0], 'original': quran_df[1],
                             'original_normalized': quran_df[1].map(quran_normalizer),
                             'lemma': quran_lemma_df[1],
                             'lemma_normalized': quran_lemma_df[1].map(quran_normalizer),
                             'root': quran_root_df[1],
                             'root_normalized': quran_root_df[1].map(quran_normalizer)},
                            axis=1).set_index('index')
merged_quran_vec_df_nrmlz = merged_quran_df[['original_normalized', 'lemma_normalized', 'root_normalized']]
# %% is this process useful?
# stopwords = [x.strip() for x in codecs.open('data/Quranic_stopwords.txt', 'r', 'utf-8').readlines()]
# stopwords = stopwords + [quran_normalizer(x.strip()) for x in
#                          codecs.open('data/Quranic_stopwords.txt', 'r', 'utf-8').readlines()]
#
#
# def delete_stopwords(sent):
#     if sent is np.nan:
#         return sent
#     return ' '.join([x for x in sent.split() if x not in stopwords])
#
#
# verse_complete_dict_nrmlz_nonstop = {k: delete_stopwords(v) for k, v in tqdm.tqdm(verse_complete_dict_nrmlz.items())}
# verse_root_dict_nrmlz_nonstop = {k: delete_stopwords(v) for k, v in tqdm.tqdm(verse_root_dict_nrmlz.items())}
# verse_lemma_dict_nrmlz_nonstop = {k: delete_stopwords(v) for k, v in tqdm.tqdm(verse_lemma_dict_nrmlz.items())}
# %%
