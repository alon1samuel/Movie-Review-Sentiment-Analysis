from datasets import load_dataset
import polars as pl
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

dataset = load_dataset("yelp_review_full")
path_to_data = Path("data/imdb.parquet")
data_df = pl.read_parquet(path_to_data)
text_column_name = "content"
label_column_name = "pos_neg"
df = (pl.DataFrame()
      .with_columns(data_df[text_column_name].alias('text'))
      .with_columns(data_df[label_column_name].alias('label'))
      .with_columns((pl.col('label')=='pos').cast(pl.Int8))
)
train_prep_df = data_df.filter(pl.col('train_test')=='train')
train, validation = train_test_split(
    train_prep_df, test_size=0.2, random_state=42
)
def transform_df_to_dataset(df):
    return Dataset.from_dict((df[[text_column_name, label_column_name]]
     .rename({text_column_name: "text", label_column_name: "label"})
      .with_columns((pl.col('label')=='pos').cast(pl.Int8))
     ).to_dict())
imdb_dataset = DatasetDict({
    "train": transform_df_to_dataset(train),
    "test": transform_df_to_dataset(validation),
})

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

imdb_dataset["train"] = imdb_dataset["train"].shuffle(seed=42).select(range(1000))
imdb_dataset["test"] = imdb_dataset["test"].shuffle(seed=42).select(range(100))
tokenized_datasets = imdb_dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"]
small_eval_dataset = tokenized_datasets["test"]
# small_train_dataset.save_to_disk('train.hf')
# small_eval_dataset.save_to_disk('eval.hf')
# from datasets import load_from_disk
# small_train_dataset = load_from_disk()
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=2
)

import numpy as np
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="test_trainer", 
    evaluation_strategy="epoch",
    learning_rate=10e-10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
