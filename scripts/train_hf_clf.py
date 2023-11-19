from datasets import load_dataset
import polars as pl
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_hf(dataset):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"]
    small_eval_dataset = tokenized_datasets["test"]
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2
    )
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


def main():
    debug=True
    path_to_data = Path("data/imdb.parquet")
    data_df = pl.read_parquet(path_to_data)
    text_column_name = "content"
    label_column_name = "pos_neg"
    df = (
        pl.DataFrame()
        .with_columns(data_df[text_column_name].alias("text"))
        .with_columns(data_df[label_column_name].alias("label"))
        .with_columns((pl.col("label") == "pos").cast(pl.Int8))
    )
    train_prep_df = data_df.filter(pl.col("train_test") == "train")
    if debug: train_prep_df = train_prep_df.sample(100, shuffle=True, seed=42)
    train, validation = train_test_split(train_prep_df, test_size=0.2, random_state=42)
    def transform_df_to_dataset(df):
        return Dataset.from_dict(
            (
                df[[text_column_name, label_column_name]]
                .rename({text_column_name: "text", label_column_name: "label"})
                .with_columns((pl.col("label") == "pos").cast(pl.Int8))
            ).to_dict()
        )
    imdb_dataset = DatasetDict(
        {
            "train": transform_df_to_dataset(train),
            "test": transform_df_to_dataset(validation),
        }
    )
    train_hf(imdb_dataset)


if __name__ == "__main__":
    metric = evaluate.load("accuracy")
    main()
