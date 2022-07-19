from preprocess_quran_text import quran_series, quran_normalizer
from tfidf_vectorizer import get_most_similars
import pandas as pd

# %%
with open('./queries.txt') as f:
    queries = f.readlines()
    queries = [q.strip() for q in queries]

results = []
i = 1
for query in queries:
    results.append({'Query': 'q{} = "{}"'.format(i, query)})
    results.extend(get_most_similars(quran_series, quran_normalizer(query), 10).to_dict('records'))
    i += 1
results = pd.DataFrame(results)
results['شباهت'] = results['شباهت'].round(3, )

