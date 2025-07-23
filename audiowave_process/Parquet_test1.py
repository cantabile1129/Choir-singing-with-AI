import panda as pd

df = pd.read_parquet("rawdata/recorded_sync.parquet")
# df['mic_signal'], df['egg_signal'], df['time_sec'] が使える
