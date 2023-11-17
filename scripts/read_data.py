import polars as pl
from pathlib import Path

DEBUG = False
path_imbd = Path("data/aclImdb")
data_path = Path("data/imdb.parquet")
files = [str(x) for x in path_imbd.rglob("*")]
if DEBUG:
    files = files[:1000]

data_list = []
for t in files:
    path_list = t.split("/")
    # skip if no train/test or pos/neg
    if len(path_list) < 5:
        continue
    if path_list[-2] == "unsup":
        continue
    with open(t) as f:
        content = f.read()
    data_list.append(
        [t, path_list[-3], path_list[-2], path_list[-1].split("_")[-1][:-4], content]
    )

data_df = pl.DataFrame(
    data_list, schema=["path", "train_test", "pos_neg", "rating", "content"]
)

data_df.write_parquet(data_path)