# ! git clone https://github.com/facebookresearch/fastText.git
# ! cd fastText
# ! make
# ! sudo pip install .

# %%
import pandas as pd
from preprocess_quran_text import quran_normalizer, merged_quran_vec_df_nrmlz, quran_series
from tools import get_most_similars
from fasttext_vectorizer import sent_to_vec, merged_corpus_embeddings

# %%
with open('./queries.txt') as f:
    queries = f.readlines()
    queries = [q.strip() for q in queries]

results = []
i = 1
for query in queries:
    query_vec = sent_to_vec(quran_normalizer(query))
    results.append({'Query': 'q{} = "{}"'.format(i, query)})
    results.extend(get_most_similars(quran_series, merged_corpus_embeddings, query_vec, 10).to_dict('records'))
    i += 1
results = pd.DataFrame(results)
results['شباهت'] = results['شباهت'].round(3, )

