import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from datasets import load_dataset

dataset = load_dataset("iwslt2017", "iwslt2017-en-de")  # English - Deutsch

train_dataset = dataset["train"]["translation"]
test_dataset = dataset["test"]["translation"]
validation_dataset = dataset["validation"]["translation"]


from tokenizers import Tokenizer, models, pre_tokenizers, trainers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer()
tokenizer.train(files=["de.txt", "en.txt"], trainer=trainer)

# Load the tokenizer
tokenizer.save("tokenizer.json")
tokenizer = Tokenizer.from_file("tokenizer.json")


class DataPreprocessor:
    def __init__(self):
        self.tokenizer = None

    def train_tokenizer(self, dataframe):
        dataframe["de"].to_csv("de.txt", index=False, header=False)
        dataframe["en"].to_csv("en.txt", index=False, header=False)

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

        trainer = trainers.BpeTrainer()
        tokenizer.train(files=["de.txt", "en.txt"], trainer=trainer)

        tokenizer.save("tokenizer.json")
        self.tokenizer = Tokenizer.from_file("tokenizer.json")

    def tokenize_data(self, df):
        df["de_tokenized"] = df["de"].apply(lambda x: self.tokenizer.encode(x).ids)
        df["en_tokenized"] = df["en"].apply(lambda x: self.tokenizer.encode(x).ids)
        return df

    def preprocess_data(self, df, train=False):
        if train or not self.tokenizer:
            self.train_tokenizer(df)
        df = self.tokenize_data(df)
        df = self.pad_data(df)
        return df

    def preprocess_data(self, df, train=False):
        if train or not self.tokenizer:
            self.train_tokenizer(df)
        df = self.tokenize_data(df)
        return df


def prepare_data(df):
    # Convert to tensors
    de_tensors = [torch.tensor(ids) for ids in df["de_tokenized"]]
    en_tensors = [torch.tensor(ids) for ids in df["en_tokenized"]]

    return de_tensors, en_tensors


def collate_fn(batch):
    de_batch = [item["de"] for item in batch]
    en_batch = [item["en"] for item in batch]
    de_padded = torch.nn.utils.rnn.pad_sequence(
        de_batch, batch_first=True, padding_value=0
    )
    en_padded = torch.nn.utils.rnn.pad_sequence(
        en_batch, batch_first=True, padding_value=0
    )
    return de_padded, en_padded


import pandas as pd

train_df = pd.DataFrame(train_dataset)
test_df = pd.DataFrame(test_dataset)
validation_df = pd.DataFrame(validation_dataset)

preprocessor = DataPreprocessor()
preprocessed_df = preprocessor.preprocess_data(train_df, train=True)

de_tensors, en_tensors = prepare_data(preprocessed_df)

data_loader = DataLoader(
    list(zip(de_tensors, en_tensors)),
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn,
)
