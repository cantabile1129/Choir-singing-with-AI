import argparse
import pandas as pd
import numpy as np
import os
import scipy.stats as stats
import scikit_posthocs as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations

def parse_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    groups = {}
    current_group = None
    i = 0
    while i < len(df):
        row = df.iloc[i]
        if isinstance(row[0], str) and row[0].strip() != "":
            current_group = row[0].strip()
            header = df.iloc[i, 1:].dropna().tolist()
            data = []
            i += 1
            while i < len(df) and pd.notna(df.iloc[i, 1]).any():
                data.append(df.iloc[i, 1:1+len(header)].tolist())
                i += 1
            groups[current_group] = pd.DataFrame(data, columns=header)
        else:
            i += 1
    return groups

def calculate_statistics(groups):
    print("📊 Calculating basic stats (mean/median/std)...")
    summary = {}
    for group_name, df in groups.items():
        df = df.apply(pd.to_numeric, errors='coerce')
        stats_df = pd.DataFrame({
            f'{group_name}_mean': df.mean(),
            f'{group_name}_median': df.median(),
            f'{group_name}_std': df.std()
        })
        summary[group_name] = stats_df
    return summary

def perform_kruskal_and_dunn(groups):
    print("📈 Performing Kruskal-Wallis and Dunn’s test...")
    questions = list(set(q for df in groups.values() for q in df.columns))
    kruskal_results = []
    dunn_results = {}
    for question in questions:
        data_for_test = []
        labels = []
        for group_name, df in groups.items():
            if question in df.columns:
                data = pd.to_numeric(df[question], errors='coerce').dropna()
                data_for_test.append(data)
                labels += [group_name] * len(data)
        if len(data_for_test) >= 2:
            H, p = stats.kruskal(*data_for_test)
            kruskal_results.append({'question': question, 'H_statistic': H, 'p_value': p})
            combined_df = pd.DataFrame({'value': pd.concat(data_for_test), 'group': labels})
            dunn = sp.posthoc_dunn(combined_df, val_col='value', group_col='group', p_adjust='bonferroni')
            dunn_results[question] = dunn
    return pd.DataFrame(kruskal_results), dunn_results

def perform_pca(groups):
    print("🧪 Performing PCA...")
    full_df = pd.concat([df.assign(group=group) for group, df in groups.items()])
    df_numeric = full_df.drop(columns=['group']).apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='any')
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_numeric)
    pca = PCA()
    pcs = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T, index=df_numeric.columns,
                            columns=[f'PC{i+1}' for i in range(len(df_numeric.columns))])
    explained_df = pd.DataFrame({'PC': [f'PC{i+1}' for i in range(len(explained))],
                                 'explained_variance_ratio': explained})
    return loadings, explained_df

def chi2_question_dependence(groups):
    print("🔗 Performing χ² tests for question independence...")
    all_questions = list(set(q for df in groups.values() for q in df.columns))
    results = []
    for q1, q2 in combinations(all_questions, 2):
        combined = []
        for df in groups.values():
            if q1 in df.columns and q2 in df.columns:
                subset = df[[q1, q2]].dropna()
                combined.append(subset)
        if combined:
            merged = pd.concat(combined)
            table = pd.crosstab(merged[q1], merged[q2])
            if table.shape[0] > 1 and table.shape[1] > 1:
                chi2, p, _, _ = stats.chi2_contingency(table)
                results.append({'question_1': q1, 'question_2': q2, 'chi2': chi2, 'p_value': p})
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    groups = parse_csv(args.csv)

    # 統計出力
    stats_summary = calculate_statistics(groups)
    merged_stats = pd.concat(stats_summary.values(), axis=1)
    merged_stats.to_csv(os.path.join(args.outdir, 'MOS_stat_summary.csv'), encoding='utf-8-sig')

    # Kruskal + Dunn
    kruskal_df, dunn_dict = perform_kruskal_and_dunn(groups)
    kruskal_df.to_csv(os.path.join(args.outdir, 'MOS_kruskal_summary.csv'), index=False, encoding='utf-8-sig')
    for q, df in dunn_dict.items():
        df.to_csv(os.path.join(args.outdir, f'MOS_dunn_{q[:20]}.csv'), encoding='utf-8-sig')

    # PCA
    pca_loadings, pca_var = perform_pca(groups)
    pca_loadings.to_csv(os.path.join(args.outdir, 'MOS_pca_loadings.csv'), encoding='utf-8-sig')
    pca_var.to_csv(os.path.join(args.outdir, 'MOS_pca_variance.csv'), index=False, encoding='utf-8-sig')

    # χ²
    chi2_df = chi2_question_dependence(groups)
    chi2_df.to_csv(os.path.join(args.outdir, 'MOS_chi2_dependence.csv'), index=False, encoding='utf-8-sig')

    print("✅ 全処理完了しました。")

if __name__ == '__main__':
    main()
