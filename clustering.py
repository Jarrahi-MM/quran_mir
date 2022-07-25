# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from quran_ir import FasttextQuranIR

sns.set()

# %%

maki = pd.read_csv('./data/مکی.csv').iloc[:, 0].apply(int).to_numpy()
madani = pd.read_csv('./data/مدنی.csv').iloc[:, 0].apply(int).to_numpy()
madani.sort()
maki.sort()

# %%
fasttext_quran_ir = FasttextQuranIR()
# X = merged_corpus_embeddings.sum(axis=1).to_frame()
X = fasttext_quran_ir.merged_corpus_embeddings[['original_normalized']]
X['شماره سوره'] = X.index.to_series().str.split('##').apply(lambda x: int(x[0]))
X = X.groupby(['شماره سوره']).sum()
X_type = pd.DataFrame(data={'نوع سوره': 'مکی'}, index=X.index)
X_type.loc[X_type.index.isin(madani), 'نوع سوره'] = 'مدنی'
X_type = X_type.reset_index()
X = np.array(X.iloc[:,0].to_list())
X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
# %% PCA
# using PCA as a dimensionality reduction transform:
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=(X_type['نوع سوره'] == 'مکی').to_list(), edgecolor='none', alpha=0.6,
            cmap=plt.cm.get_cmap('Spectral', 10))

plt.show()
# %% # t-SNE
# Fit and transform with a TSNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
            c=(X_type['نوع سوره'] == 'مکی').to_list(), edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))

plt.show()
# %% K-means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0, n_init=200, max_iter=1000).fit(X)
kmeans_labels = kmeans.labels_

# %%
# RSS , the squared distance of each vector from its centroid summed over all vectors
centers = kmeans.cluster_centers_[kmeans_labels]
rss = np.sum((X - centers) ** 2, axis=1).sum()
print(f'{"RSS manual:":<80} {rss:.4f}')
print(f'{"RSS auto:":<80} {-kmeans.score(X):.4f}')

contingency_matrix = sklearn.metrics.cluster.contingency_matrix(X_type['نوع سوره'] == 'مدنی', kmeans_labels)
y_true, y_pred = X_type["نوع سوره"] == "مدنی", kmeans_labels
print(f'{"Purity:":<80} {np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix):.4f}')
print(f'{"accuracy_score:":<80} {sklearn.metrics.accuracy_score(y_true, y_pred):.4f}')
print(f'{"balanced_accuracy_score:":<80} {sklearn.metrics.balanced_accuracy_score(y_true, y_pred):.4f}')
print(
    f'{"fowlkes_mallows_score (The score ranges from 0 to 1):":<80} {sklearn.metrics.fowlkes_mallows_score(y_true, y_pred):.4f}')
print(f'{"rand_score (Score between 0.0 and 1.0):":<80} {sklearn.metrics.rand_score(y_true, y_pred):.4f}')
# %%
plt.figure(figsize=(14,14))

selection = (X_type['نوع سوره'] == 'مکی')
plt.scatter(X_pca[selection, 0], X_pca[selection, 1],
            c=(kmeans_labels[selection] == 0), edgecolor='none', alpha=0.6, s=200,
            marker='*',
            cmap=plt.cm.get_cmap('Spectral', 10))

selection = (X_type['نوع سوره'] == 'مدنی')
plt.scatter(X_pca[selection, 0], X_pca[selection, 1],
            c=(kmeans_labels[selection] == 1), edgecolor='none', alpha=0.6, s=100,
            marker=',',
            cmap=plt.cm.get_cmap('Spectral', 10))

centers_low_dim=pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_low_dim[:,0], centers_low_dim[:,1],
            c='orange', edgecolor='none', alpha=0.6, s=100,
            marker=',')

plt.show()
