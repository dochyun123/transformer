# -*- coding: utf-8 -*-
"""study_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n_a_m0Wb44NQiAyxpGx3mNjidwR92qlp
"""

# pip install datasets

from datasets import load_dataset

dataset = load_dataset("iwslt2017", "iwslt2017-en-de")  # English - Deutsch

print(len(dataset["train"]["translation"]))
print(len(dataset["test"]["translation"]))
print(len(dataset["validation"]["translation"]))

train_dataset = dataset["train"]["translation"]
test_dataset = dataset["test"]["translation"]
validation_dataset = dataset["validation"]["translation"]

import pandas as pd

train_df = pd.DataFrame(train_dataset)
test_df = pd.DataFrame(test_dataset)
validation_df = pd.DataFrame(validation_dataset)

# pip install tokenizers

train_df.describe()

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

train_df["de"].to_csv("de.txt", index=False, header=False)
train_df["en"].to_csv("en.txt", index=False, header=False)

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"])
tokenizer.train(files=["de.txt", "en.txt"], trainer=trainer)

# Load the tokenizer
tokenizer.save("tokenizer.json")
tokenizer = Tokenizer.from_file("tokenizer.json")

# Tokenize the data
train_df["de_tokenized"] = train_df["de"].apply(lambda x: tokenizer.encode(x).ids)
train_df["en_tokenized"] = train_df["en"].apply(lambda x: tokenizer.encode(x).ids)

sample_sentence = train_df["de"][10000]
tokenized = tokenizer.encode(sample_sentence)
decoded = tokenizer.decode(tokenized.ids)
print(f"Original: {sample_sentence}")
print(f"Tokenized IDs: {tokenized.ids}")
print(f"Decoded: {decoded}")


def pad_sequence(sequence, max_length):
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        return sequence + [0] * (max_length - len(sequence))


max_length = 100
train_df["de_padded"] = train_df["de_tokenized"].apply(
    lambda x: pad_sequence(x, max_length)
)
train_df["en_padded"] = train_df["en_tokenized"].apply(
    lambda x: pad_sequence(x, max_length)
)

start_token = tokenizer.token_to_id("[CLS]")
end_token = tokenizer.token_to_id("[SEP]")

train_df["de_padded"] = train_df["de_padded"].apply(
    lambda x: [start_token] + x + [end_token]
)
train_df["en_padded"] = train_df["en_padded"].apply(
    lambda x: [start_token] + x + [end_token]
)

train_df.to_csv("train.csv", index=False)

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


class DataPreprocessor:
    def __init__(
        self, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"], max_length=100
    ):
        self.tokenizer = None
        self.special_tokens = special_tokens
        self.max_length = max_length

    def train_tokenizer(self, dataframe):
        dataframe["de"].to_csv("de.txt", index=False, header=False)
        dataframe["en"].to_csv("en.txt", index=False, header=False)

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

        trainer = trainers.BpeTrainer(special_tokens=self.special_tokens)
        tokenizer.train(files=["de.txt", "en.txt"], trainer=trainer)

        tokenizer.save("tokenizer.json")
        self.tokenizer = Tokenizer.from_file("tokenizer.json")

    def tokenize_data(self, df):
        df["de_tokenized"] = df["de"].apply(lambda x: self.tokenizer.encode(x).ids)
        df["en_tokenized"] = df["en"].apply(lambda x: self.tokenizer.encode(x).ids)
        return df

    def pad_sequence(self, sequence):
        return (
            sequence[: self.max_length]
            if len(sequence) > self.max_length
            else sequence + [0] * (self.max_length - len(sequence))
        )

    def pad_data(self, df):
        start_token = self.tokenizer.token_to_id("[CLS]")
        end_token = self.tokenizer.token_to_id("[SEP]")

        df["de_padded"] = df["de_tokenized"].apply(
            lambda x: [start_token] + self.pad_sequence(x) + [end_token]
        )
        df["en_padded"] = df["en_tokenized"].apply(
            lambda x: [start_token] + self.pad_sequence(x) + [end_token]
        )
        return df

    def preprocess_data(self, df, train=False):
        if train or not self.tokenizer:
            self.train_tokenizer(df)
        df = self.tokenize_data(df)
        df = self.pad_data(df)
        return df


# Using the class
preprocessor = DataPreprocessor(max_length=100)
train_df = preprocessor.preprocess_data(train_df, train=True)
test_df = preprocessor.preprocess_data(test_df)
validation_df = preprocessor.preprocess_data(validation_df)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
validation_df.to_csv("validation.csv", index=False)
