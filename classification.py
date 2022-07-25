# %%
import numpy as np
from quran_ir import FasttextQuranIR

fasttext_quran_ir = FasttextQuranIR()
X = fasttext_quran_ir.merged_corpus_embeddings[['original_normalized']].copy()
X['شماره سوره'] = X.index.to_series().str.split('##').apply(lambda x: int(x[0]))
X['شماره آیه'] = X.index.to_series().str.split('##').apply(lambda x: int(x[1]))

long_verses = X.groupby(['شماره سوره']).count()['شماره آیه'].sort_values(ascending=False)[:30].index
X_long = X[X['شماره سوره'].isin(long_verses)]
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(np.array(X_long['original_normalized'].to_list()),
                                                    np.array(X_long['شماره سوره'].to_list()), test_size=0.2,
                                                    random_state=1)
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

# clf = LogisticRegression(random_state=0, n_jobs=4).fit(X_train, Y_train)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 60, 30), random_state=1,
                    max_iter=10000).fit(X_train, Y_train)
y_predicted = clf.predict(X_test)

print(f'accuracy_score: {accuracy_score(Y_test, y_predicted)}')
print(f'macro f1_score: {f1_score(Y_test, y_predicted, average="macro")}')
print(f'micro f1_score: {f1_score(Y_test, y_predicted, average="micro")}')
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess_quran_text import merged_quran_vec_df_nrmlz

vec = TfidfVectorizer(norm='l2').fit(merged_quran_vec_df_nrmlz['original_normalized'])

X = merged_quran_vec_df_nrmlz.copy()
X['شماره سوره'] = X.index.to_series().str.split('##').apply(lambda x: int(x[0]))
X['شماره آیه'] = X.index.to_series().str.split('##').apply(lambda x: int(x[1]))

long_verses = X.groupby(['شماره سوره']).count()['شماره آیه'].sort_values(ascending=False)[:30].index
X_long = X[X['شماره سوره'].isin(long_verses)]
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(vec.transform(X_long['original_normalized']),
                                                    np.array(X_long['شماره سوره'].to_list()), test_size=0.2,
                                                    random_state=1)
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf = LogisticRegression(random_state=0, n_jobs=4).fit(X_train, Y_train)
y_predicted = clf.predict(X_test)

print(f'accuracy_score: {accuracy_score(Y_test, y_predicted)}')
print(f'macro f1_score: {f1_score(Y_test, y_predicted, average="macro")}')
print(f'micro f1_score: {f1_score(Y_test, y_predicted, average="micro")}')
