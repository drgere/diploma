import pandas as pd

defmod_metrics = ['MvSc.', 'S-BLEU', 'L-BLEU']
lng = ['EN', 'ES', 'FR', 'IT', 'RU']

defmod_results = '/home/olygina/code/semeval2022/rank/submission_scores/res_defmod.csv'


def orig_rankings(resfile):
    df = pd.read_csv(resfile)

    def get_sorted_vals(colname):
        return sorted(df[colname].dropna(), reverse=True)

    for colname in [f"{lang} {metric}" for lang in lng for metric in defmod_metrics]:
        sorted_vals = get_sorted_vals(colname)

        def float_to_rank(cell):
            if pd.isna(cell): return cell
            return sum(i >= cell for i in sorted_vals)

        df[colname] = df[colname].apply(float_to_rank)

    df.to_csv('defmod_results.csv', index=False)
    df_ranks = df.groupby('user').min()

    for lang in lng:
        def get_mean_rank(row):
            metrics = [row[f"{lang} {metric}"] for metric in defmod_metrics]
            if any(map(pd.isna, metrics)): return pd.NA
            return sum(metrics) / len(metrics)

        df_ranks[f"Rank {lang}"] = df_ranks.apply(get_mean_rank, axis=1)

    df_ranks.to_csv('defmod_rank_users.csv')


def scores_personal(orig_res_file, user='olygina', outfile=None):
    df_orig = pd.read_csv(orig_res_file)
    udf = df_orig[df_orig['user'] == user].copy()
    udf.drop(columns=['Comments', 'Date', 'user'], inplace=True)
    if not outfile:
        outfile = f'{user}_scores.csv'
    udf.to_csv(outfile)


if __name__ == '__main__':
    scores_personal(defmod_results)
