import pandas as pd

# 入力CSVファイルのパス
csv_file = "recordings/pre_id2_solo.csv"

# 出力Parquetファイルのパス
parquet_file = "recordings/pre_id2_solo.parquet"

# CSVを読み込み
df = pd.read_csv(csv_file)

# Parquet形式で保存（圧縮付き）
df.to_parquet(parquet_file, engine="pyarrow", compression="snappy", index=False)

print(f"CSVファイル '{csv_file}' を Parquet '{parquet_file}' に変換しました！")

