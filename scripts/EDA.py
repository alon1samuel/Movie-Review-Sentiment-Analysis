import polars as pl
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.decomposition import PCA


DEBUG = True
path_imbd = Path("data/aclImdb")
test_dir = Path("data/aclImdb/test")
train_dir = Path("data/aclImdb/train")
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

# train = data_df.filter(pl.col('train_test')=='train')


def cluster_and_label(dataframe, num_clusters=3, column_name="content"):
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()
    # Fit and transform the 'content' column
    tfidf_matrix = vectorizer.fit_transform(dataframe[column_name])
    pca = PCA(n_components=2)
    tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())

    # Initialize KMeans with the specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # Fit KMeans to the TF-IDF matrix
    kmeans.fit(tfidf_matrix)
    # Add a 'label' column to the DataFrame indicating the cluster assignment
    return (dataframe
            .with_columns(pl.Series(kmeans.labels_).alias("label"))
            .with_columns(pl.Series(tfidf_pca[:, 0]).alias("pca_x"))
            .with_columns(pl.Series(tfidf_pca[:, 1]).alias("pca_y"))
            )


clustered_df = cluster_and_label(data_df, column_name="content")

fig = px.scatter(
    clustered_df,
    x="pca_x",
    y="pca_y",
    color="label",
    symbol="label",
    size_max=10,
    opacity=0.7,
    title="K-means Clustering of TF-IDF",
)
fig.show()


fig = px.scatter(
    clustered_df,
    x="pca_x",
    y="pca_y",
    color="pos_neg",
    symbol="pos_neg",
    size_max=10,
    opacity=0.7,
    title="K-means Clustering of TF-IDF",
)
fig.show()

clustered_df.describe()

print()
