import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }


class EntityModel:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3)

    def preprocess_data(self, filepath):
        data = pd.read_csv(filepath, delimiter=';')
        texts = data['text'].tolist()

        # генерация меток на уровне символов
        labels = []
        for _, row in data.iterrows():
            text = row['text']
            account = str(row['personal_account_number'])
            meter = str(row['meter_readings'])

            char_labels = np.zeros(len(text), dtype=int)

            # помечаем позиции лицевого счета
            if account in text:
                start = text.index(account)
                end = start + len(account)
                char_labels[start:end] = 1

            # помечаем позиции показаний счетчика
            if meter in text:
                start = text.index(meter)
                end = start + len(meter)
                char_labels[start:end] = 2

            labels.append(char_labels.tolist())

        return texts, labels

    def train(self, texts, labels):
        # токенизация с выравниванием меток
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        # выравнивание меток для токенов
        aligned_labels = []
        for i, (label, offset) in enumerate(zip(labels, tokenized.offset_mapping)):
            label_ids = []
            for token_idx, (start, end) in enumerate(offset):
                if start == end:
                    label_ids.append(-100)
                else:
                    # берем метку первого символа токена
                    label_ids.append(label[start])
            aligned_labels.append(label_ids)

        labels_tensor = torch.tensor(aligned_labels)

        dataset = CustomDataset(tokenized, labels_tensor)

        dataset_size = len(dataset)
        indices = torch.randperm(dataset_size).tolist()
        train_size = int(0.8 * dataset_size)

        train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
        eval_dataset = torch.utils.data.Subset(dataset, indices[train_size:])

        # параметры обучения
        training_args = TrainingArguments(
            output_dir='results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='logs',
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        trainer.train()

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)