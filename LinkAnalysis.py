# %%
import matplotlib.pyplot as plt
import pandas as pd
from preprocess_quran_text import merged_quran_vec_df_nrmlz, quran_series
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize
import warnings
import tqdm
from quran_ir import FasttextQuranIR

warnings.filterwarnings("ignore")

# %%
verse_names = pd.read_csv('data/verse_names.csv')
verse_names.set_index(verse_names['ردیف'], inplace=True)

# %%
fasttext_quran_ir = FasttextQuranIR()
# %%
X = fasttext_quran_ir.merged_corpus_embeddings[['original_normalized']]
X['شماره سوره'] = X.index.to_series().str.split('##').apply(lambda x: int(x[0]))
# %%
list_of_verse_embeddings = [x for _, x in X.groupby(['شماره سوره'])]


# %%
def plot_graph(G):
    max_d = np.max([d["weight"] for (u, v, d) in G.edges(data=True)])
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == max_d]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] != max_d]

    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=5)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=3, alpha=0.5, edge_color="r", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# %%
def find_pivot_aye(verse_name, draw_graph=False):
    verse_number = verse_names.loc[verse_names['نام سوره'] == verse_name]['ردیف'].tolist()[0]
    df = list_of_verse_embeddings[verse_number - 1]
    verse_matrix = np.array(df['original_normalized'].values.tolist())

    P = verse_matrix.dot(verse_matrix.T)
    np.fill_diagonal(P, 0)
    P_norm = normalize(P, norm='l1')
    G = nx.from_numpy_matrix(P_norm, create_using=nx.MultiDiGraph())
    pr = nx.pagerank(G, alpha=0.9)
    h, a = nx.hits(G)
    if draw_graph:
        plot_graph(G)
    pivot_aye_pr = quran_series[df.index[np.argmax(list(pr.values()))]]
    pivot_aye_a = quran_series[df.index[np.argmax(list(a.values()))]]
    pivot_aye_h = quran_series[df.index[np.argmax(list(h.values()))]]
    return pd.DataFrame(
        {'verse_name': verse_name, 'pivot_aye_pr': pivot_aye_pr, 'pivot_aye_a': pivot_aye_a,
         'pivot_aye_h': pivot_aye_h},
        index=[df['شماره سوره'][0]])


# %%
print(find_pivot_aye('کوثر', draw_graph=True))
# print(find_pivot_aye('فاتحه', draw_graph=True))
# print(find_pivot_aye('قدر', draw_graph=True))
# print(find_pivot_aye('کافرون'))
# print(find_pivot_aye('فیل'))
# print(find_pivot_aye('نصر'))
# print(find_pivot_aye('فلق'))
# print(find_pivot_aye('شمس'))
# print(find_pivot_aye('ناس' , draw_graph=True))

# %%
pivot_aye = pd.DataFrame()
for index, row in tqdm.tqdm(verse_names.iterrows()):
    pivot_aye = pd.concat([pivot_aye, find_pivot_aye(row['نام سوره'])])
pivot_aye
# %%
print(pivot_aye['pivot_aye_a'].equals(pivot_aye['pivot_aye_h']))
print(pivot_aye['pivot_aye_a'].equals(pivot_aye['pivot_aye_pr']))
print(pivot_aye.loc[pivot_aye['pivot_aye_a'] != pivot_aye['pivot_aye_pr']])
print(pivot_aye.loc[pivot_aye['pivot_aye_a'] == pivot_aye['pivot_aye_h']])
print(pivot_aye.loc[pivot_aye['pivot_aye_h'] == pivot_aye['pivot_aye_pr']])

# %%
