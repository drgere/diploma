import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP

from analytics.utils import loading_embeddings

def embedding_plot(dset='dset_v1', lang='en', emb='sgns', reduce_method='TSNE',
                   rndSeed=81773, subsample=None):
    test_embedding = loading_embeddings(lang, subset='test', emb=emb, subdir='defmod-test', label='defmod')
    train_embedding = loading_embeddings(lang, subset='train', emb=emb, subdir=dset)
    test_len, train_len = len(test_embedding), len(train_embedding)
    if subsample:
        np.random.seed(rndSeed)
        test_embedding = test_embedding[choice(test_len, int(test_len * subsample), replace=False), :]
        test_len = int(test_len*subsample)
        train_embedding = train_embedding[choice(train_len, int(train_len * subsample), replace=False), :]
        train_len = int(train_len * subsample)
    all_embeddings = np.concatenate((train_embedding, test_embedding))
    if reduce_method == 'TSNE': dimReduce = TSNE(random_state=rndSeed)
    elif reduce_method == 'UMAP': dimReduce = UMAP(random_state=rndSeed)
    embedding_2D = dimReduce.fit_transform(all_embeddings)
    color = np.concatenate((np.ones(train_len) * 0.7, np.ones(test_len) * 0.3))
    fig, ax = plt.subplots();
    ax.scatter(embedding_2D[:, 0], embedding_2D[:, 1], s=0.5, c=color,
               alpha=0.3, cmap='RdYlGn')
    figid=f'trainVsTestDensity{reduce_method}-{lang}-{emb}'
    plt.axis('off')
    plt.lay_out(pad=0)
    plt.savefig(figid+'.png', dpi=300)

def all_plots():
    for lang in ['en', 'es', 'fr', 'it', 'ru']:
        vectors = ['sgns', 'electra'] if lang in ['en', 'fr', 'ru'] else ['sgns']
        for vec in vectors:
            print(lang, vec)
            embedding_plot(lang=lang, emb=vec, reduce_method='TSNE', subsample=None)

if __name__ == '__main__':
    all_plots()