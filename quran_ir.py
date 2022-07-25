from abc import abstractmethod
from preprocess_quran_text import merged_quran_vec_df_nrmlz, quran_normalizer
import pandas as pd
import numpy as np
from preprocess_quran_text import quran_series


class QuranIR:

    def __init__(self):
        self.algorithm = '?'

    @abstractmethod
    def get_most_similars(self, query: str, K=10, check_moghattaeh=False) -> pd.DataFrame:
        raise NotImplementedError("Please Implement this method")

    def process_queries(self, query_path='./queries.txt', result_path='ir_responses/') -> pd.DataFrame:
        with open(query_path) as f:
            queries = f.readlines()
            queries = [q.strip() for q in queries]

        results = []
        i = 1
        for query in queries:
            results.append({'Query': 'q{} = "{}"'.format(i, query)})
            results.extend(self.get_most_similars(quran_normalizer(query), 10).to_dict('records'))
            i += 1
        results = pd.DataFrame(results)
        results['شباهت'] = results['شباهت'].round(3, )
        results.to_csv(f'{result_path}{self.algorithm}.csv', index=False)
        return results


class BooleanQuranIR(QuranIR):

    def __init__(self):
        from preprocess_quran_text import verse_complete_dict_nrmlz, verse_lemma_dict_nrmlz, verse_root_dict_nrmlz, \
            verse_complete_dict
        from boolean_retrieval.ir_system import IRSystem
        super().__init__()
        self.algorithm = 'Boolean'
        # TODO: use merged_quran_df
        self.keys = [*verse_complete_dict.keys()]
        self.docs, self.docs_complete, self.docs_lemma, self.docs_root = [*verse_complete_dict.values()], [
            *verse_complete_dict_nrmlz.values()], [*verse_lemma_dict_nrmlz.values()], [*verse_root_dict_nrmlz.values()]
        self.boolean_ir_complete, self.boolean_ir_lemma, self.boolean_ir_root = IRSystem(self.docs_complete), IRSystem(
            self.docs_lemma), IRSystem(self.docs_root)

    def get_most_similars(self, query: str, K=10, check_moghattaeh=False) -> pd.DataFrame:
        result = self.boolean_ir_complete.process_query(query.split(), "complete")
        results_lemma = self.boolean_ir_lemma.process_query(query.split(), "lemma")
        results_root = self.boolean_ir_root.process_query(query.split(), "root")
        result.extend([r for r in results_lemma if r not in result])
        result.extend([r for r in results_root if r not in result])
        result = pd.DataFrame(data={'آیه': [self.docs[r] for r in result]}, index=[self.keys[r] for r in result])
        return result.iloc[:K]

    def process_queries(self, query_path='./queries.txt', result_path='ir_responses/') -> pd.DataFrame:
        with open('./queries_boolean.txt') as f:
            queries = f.readlines()
            queries = [q.strip() for q in queries]

        results = []
        i = 1
        for query in queries:
            results.append({'Query': 'q{} = "{}"'.format(i, query)})
            results.extend(self.get_most_similars(query).to_dict('records'))
            i += 1

        results = pd.DataFrame(results)
        results.to_csv(f'{result_path}{self.algorithm}.csv', index=False)
        return results


class TfIdfQuranIR(QuranIR):

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        super().__init__()
        self.algorithm = 'TfIdf'
        corpus = merged_quran_vec_df_nrmlz.fillna(' ').to_numpy().flatten('F').tolist()
        self.vectorizer = TfidfVectorizer(norm='l2')
        self.doc_term_matrix = self.vectorizer.fit_transform(corpus)

        self.words = self.vectorizer.get_feature_names()
        self.idf_mat = self.vectorizer.idf_
        self.median_idf = (np.max(self.idf_mat) + np.min(self.idf_mat)) / 2

    def get_most_similars(self, query: str, K=10, check_moghattaeh=False) -> pd.DataFrame:
        from sklearn.metrics.pairwise import linear_kernel
        embedded_query = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(self.doc_term_matrix, embedded_query).flatten()

        quran_len = len(quran_series)
        similarity_df = pd.DataFrame(data={'original_normalized': cosine_similarities[0:quran_len],
                                           'lemma_normalized': cosine_similarities[quran_len:2 * quran_len],
                                           'root_normalized': cosine_similarities[2 * quran_len:],
                                           },
                                     index=quran_series.index)
        similarity_series = pd.concat(
            [similarity_df['original_normalized'], similarity_df['root_normalized'], similarity_df['lemma_normalized']],
            axis=1).max(axis=1)
        selected_ayats = similarity_series.sort_values(ascending=False)[:K]

        return pd.DataFrame(data={'آیه': quran_series[selected_ayats.index],
                                  'شباهت': selected_ayats}, index=selected_ayats.index)

    def get_word_idf(self, word):
        # should be normalize_and_delete_stopwords
        if word not in self.words:
            return self.median_idf
        return self.idf_mat[self.words.index(word)]


class FasttextQuranIR(QuranIR):
    EMBEDDING_LEN = 100

    def __init__(self):
        import fasttext
        super().__init__()
        self.algorithm = 'Fasttext'
        self.model = fasttext.load_model('fasttext_model/model.bin')
        self.tfidf_quran_ir = TfIdfQuranIR()
        self.merged_corpus_embeddings = merged_quran_vec_df_nrmlz.applymap(self.sent_to_vec)

    @staticmethod
    def train(create_dataset=False):
        # ! git clone https://github.com/facebookresearch/fastText.git
        # ! cd fastText
        # ! make
        # ! sudo pip install .
        from tools import create_data_set
        import subprocess
        import shlex
        if create_dataset:
            create_data_set(out_dir='./fasttext_model/data_set.txt', merged_df=merged_quran_vec_df_nrmlz,
                            expansion_count=5, lemma_rate=0.1, root_rate=0.4)

        command = "./fastText/fasttext skipgram -input ./fasttext_model/data_set.txt -output ./fasttext_model/model " \
                  "-ws 5 -dim 100 -minn 3 -maxn 10 -epoch 1000 -thread 15 "
        subprocess.run(shlex.split(command))

    def sent_to_vec(self, sent: str):
        if pd.isna(sent):
            return np.zeros(FasttextQuranIR.EMBEDDING_LEN)
        words = sent.split()
        if len(words) == 0:
            return np.zeros(FasttextQuranIR.EMBEDDING_LEN)
        vec = np.average(a=[self.model.get_word_vector(word) for word in words],
                         weights=[self.tfidf_quran_ir.get_word_idf(word) for word in words],
                         axis=0)
        return vec / np.linalg.norm(vec)

    def get_most_similars(self, query: str, K=10, check_moghattaeh=False) -> pd.DataFrame:
        import tools
        return tools.get_most_similars(merged_corpus_embeddings=self.merged_corpus_embeddings,
                                       query_vec=self.sent_to_vec(query),
                                       K=K,
                                       check_moghattaeh=check_moghattaeh)


class ArabertQuranIR(QuranIR):
    EMBEDDING_LEN = 768

    def __init__(self):
        from arabert.preprocess import ArabertPreprocessor
        from transformers import AutoTokenizer, AutoModel
        super().__init__()
        self.algorithm = 'Arabert'
        self.tfidf_quran_ir = TfIdfQuranIR()
        model_name = "aubmindlab/bert-base-arabertv2"
        self.arabert_prep = ArabertPreprocessor(model_name=model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.count = 0
        # merged_quran_df or merged_quran_vec_df_nrmlz
        self.merged_corpus_embeddings = merged_quran_vec_df_nrmlz.applymap(self.sent_to_vec)

    def sent_to_vec(self, sent: str):
        if sent == '':
            return np.zeros(ArabertQuranIR.EMBEDDING_LEN)
        text_preprocessed = self.arabert_prep.preprocess(sent)
        arabert_input = self.tokenizer.encode_plus(text_preprocessed, return_tensors='pt')
        tokens = self.tokenizer.convert_ids_to_tokens(arabert_input['input_ids'][0])[1:-1]
        outputs = self.model(**arabert_input)
        embeddings_text_only = outputs['last_hidden_state'][0][1:-1]
        self.count += 1
        if self.count % 1000 == 0:
            print(self.count)
        avg_vec = np.average(a=embeddings_text_only.detach().numpy(), weights=[self.tfidf_quran_ir.get_word_idf(
            quran_normalizer(word)) if '+' not in word else 0 for word in tokens], axis=0)
        if np.linalg.norm(avg_vec) == 0:
            return np.zeros(ArabertQuranIR.EMBEDDING_LEN)
        return avg_vec / np.linalg.norm(avg_vec)

    def get_most_similars(self, query: str, K=10, check_moghattaeh=True) -> pd.DataFrame:
        import tools
        return tools.get_most_similars(merged_corpus_embeddings=self.merged_corpus_embeddings,
                                       query_vec=self.sent_to_vec(query),
                                       K=K,
                                       check_moghattaeh=check_moghattaeh)
