# %% import section
from boolean_retrieval.ir_system import IRSystem
from preprocess_quran_text import verse_complete_dict_nrmlz, verse_lemma_dict_nrmlz, \
    verse_root_dict_nrmlz, verse_complete_dict
import pandas as pd

# %% initialize IR system
docs, docs_complete, docs_lemma, docs_root = [*verse_complete_dict.values()], [*verse_complete_dict_nrmlz.values()], [
    *verse_lemma_dict_nrmlz.values()], [*verse_root_dict_nrmlz.values()]
boolean_ir_complete, boolean_ir_lemma, boolean_ir_root = IRSystem(docs_complete), IRSystem(docs_lemma), IRSystem(
    docs_root)

# %%
k = 10

with open('./queries_boolean.txt') as f:
    queries = f.readlines()
    queries = [q.strip().split() for q in queries]

results = []
i = 1
for query in queries:
    result = boolean_ir_complete.process_query(query, "complete")
    results_lemma = boolean_ir_lemma.process_query(query, "lemma")
    results_root = boolean_ir_root.process_query(query, "root")
    result.extend([r for r in results_lemma if r not in result])
    result.extend([r for r in results_root if r not in result])
    result = [docs[r] for r in result]
    results.append('q{} = "{}"'.format(i, ' '.join(query)))
    results.extend(result[0:k])
    i += 1

results = pd.DataFrame(results)
