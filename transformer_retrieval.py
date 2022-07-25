# %%
# !pip install transformers
# !git clone https://github.com/aub-mind/arabert
# !pip install -r arabert/requirements.txt
# %%
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor
from preprocess_quran_text import quran_series, quran_normalizer, merged_quran_vec_df_nrmlz
from tools import get_most_similars
import numpy as np
import pandas as pd
from quran_ir import TfIdfQuranIR

# %%
tfidf_quran_ir = TfIdfQuranIR()
# %%
EMBEDDING_LEN = 768
model_name = "aubmindlab/bert-base-arabertv2"
arabert_prep = ArabertPreprocessor(model_name=model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
count = 0


def sent_to_vec(sent):
    global count
    if sent == '':
        return np.zeros(EMBEDDING_LEN)
    text_preprocessed = arabert_prep.preprocess(sent)
    arabert_input = tokenizer.encode_plus(text_preprocessed, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(arabert_input['input_ids'][0])[1:-1]
    outputs = model(**arabert_input)
    embeddings_text_only = outputs['last_hidden_state'][0][1:-1]
    count += 1
    if count % 1000 == 0:
        print(count)
    avg_vec = np.average(a=embeddings_text_only.detach().numpy(), weights=[tfidf_quran_ir.get_word_idf(
        quran_normalizer(word)) if '+' not in word else 0 for word in tokens], axis=0)
    if np.linalg.norm(avg_vec) == 0:
        return np.zeros(EMBEDDING_LEN)
    return avg_vec / np.linalg.norm(avg_vec)


# %%
# merged_quran_df or merged_quran_vec_df_nrmlz
merged_corpus_embeddings = merged_quran_vec_df_nrmlz.applymap(sent_to_vec)
# %%

with open('./queries.txt') as f:
    queries = f.readlines()
    queries = [q.strip() for q in queries]

results = []
i = 1
for query in queries:
    query_vec = sent_to_vec(quran_normalizer(query))
    results.append({'Query': 'q{} = "{}"'.format(i, query)})
    results.extend(
        get_most_similars(quran_series, merged_corpus_embeddings, query_vec, 10, check_moghattaeh=True).to_dict(
            'records'))
    i += 1
results = pd.DataFrame(results)
results['شباهت'] = results['شباهت'].round(3, )

# %%
# query = 'وَلِلّهِ الأَسْمَاء الْحُسْنَى'
#
# query_vec = sent_to_vec(quran_normalizer(query))
# r = get_most_similars(quran_series, merged_corpus_embeddings, query_vec, 10, check_moghattaeh=True)
# print(r)
