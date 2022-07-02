import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from analytics.utils import loading_embeddings, loading_json

submit_best = {lang: f'/home/olygina/codwoe/defmod-analysis/submitV4-lstm-gru-allvec/{lang}.modelout.alldata.json'
               for lang in ['en', 'es', 'fr', 'it', 'ru']}
best = {lang: f'/home/olygina/codwoe/defmod-analysis/submitV4-gru-allvec-sgns/{lang}.modelout.alldata.json'
        for lang in ['en', 'es', 'fr', 'it', 'ru']}

sim_best = lambda x: x[0]
avg_best_10 = lambda x: np.average(x[:10])
avg_best_20 = lambda x: np.average(x[:20])
avg_best_50 = lambda x: np.average(x[:50])
med_best_10 = lambda x: np.median(x[:10])
med_est_20_q75 = lambda x: np.percentile(x[20], 75)

agg_sim = {
            'best': sim_best,
            'top10avg': avg_best_10,
            'top10med': med_best_10,
            'top20q75': med_est_20_q75
          }

def agg_percent(perc):
    return lambda x: np.sum(x >= perc)

def correlation(lang='en', train_dset='dset_v1', emb_type='sgns',
                                   closeness='sim'):
    print(f'CORRELATION FOR: {lang}, {emb_type}')
    train_embadding = loading_embeddings(lang, subset='train', emb=emb_type, subdir=train_dset)
    model_out = loading_json(model_out)
    model_out = sorted(model_out, key=lambda r: r["id"])
    test_embadding = np.stack([np.array(itm[emb_type]) for itm in model_out])
    cos_sim = cosine_similarity(test_embadding, train_embadding)
    cos_sim = -np.sort(-cos_sim, axis=1)
    best_25_cos, best_20_cos, best_10_cos = np.percentile(cos_sim.flatten(), [75, 80, 90])
    for score in ['moverscore', 'sense-BLEU', 'lemma-BLEU']:
        score = [itm[score] for itm in model_out]
        if closeness == 'sim': aggFuncs = agg_sim
        elif closeness == 'percentile':
            aggFuncs = {
                'top25cos': agg_percent(best_25_cos),
                'top20cos': agg_percent(best_20_cos),
                'top10cos': agg_percent(best_10_cos),
            }
        for sim in aggFuncs.keys():
            sima = aggFuncs[sim]
            sims = [sima(cos_sim[ix]) for ix in range(len(model_out))]
            corr, pval = spearmanr(score, sims)
            print(f'score {score:10} ; sim {sim:8} : corr={corr:.4f} pval={pval:.4f}')
    print()

def average_score(lang='en', train_dset='dset_v1', emb_type='sgns'):
    print(f'CORRELATION FOR: {lang}, {emb_type}')
    train_embadding = loading_embeddings(lang, subset='train', emb=emb_type, subdir=train_dset)
    model_out = loading_json(model_out)
    model_out = sorted(model_out, key=lambda r: r["id"])
    test_embadding = np.stack([np.array(itm[emb_type]) for itm in model_out])
    cos_sim = cosine_similarity(test_embadding, train_embadding)
    cos_sim = -np.sort(-cos_sim, axis=1)
    Ln = len(model_out)
    for score in ['moverscore', 'sense-BLEU', 'lemma-BLEU']:
        scores = [itm[score] for itm in model_out]
        for sim in agg_sim.keys():
            sima = agg_sim[sim]
            sims = [sima(cos_sim[ix]) for ix in range(Ln)]
            for perc in [10, 20, 25]:
                bottom, top = np.percentile(sims, [perc, 100-perc])
                best_score = np.average([scores[ix] for ix in range(Ln) if sims[ix] >= top])
                score = np.average([scores[ix] for ix in range(Ln) if sims[ix] <= bottom])
                print(f'score {score:10} ; sim {sim:8} : low{perc}%={score:.4f} top{perc}%={best_score:.4f}')
    print()

def scores_run(lang2outfile):
    for lang in ['en', 'es', 'fr', 'it', 'ru']:
        vec = ['sgns', 'electra'] if lang in ['en', 'fr', 'ru'] else ['sgns']
        for vec in vec:
            correlation(lang2outfile[lang], lang=lang, train_dset='dset_v1', emb_type=vec, closeness='percentile')

def avgs_run(lang2outfile):
    for lang in ['en', 'es', 'fr', 'it', 'ru']:
        vec = ['sgns', 'electra'] if lang in ['en', 'fr', 'ru'] else ['sgns']
        for vec in vec:
            average_score(lang2outfile[lang], lang=lang, train_dset='dset_v1', emb_type=vec)

if __name__ == '__main__':
    avgs_run(submit_best)