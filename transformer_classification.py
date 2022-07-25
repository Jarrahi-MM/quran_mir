from preprocess_quran_text import merged_quran_vec_df_nrmlz
import pandas as pd
import numpy as np
import pandas as pd
import qalsadi.lemmatizer
from nltk.stem.isri import ISRIStemmer
import tqdm

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_metric

from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import AutoTokenizer, DataCollatorWithPadding, pipeline, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer

import pandas as pd
import hazm
import torch
import os


stemmer = ISRIStemmer()
lemmer = qalsadi.lemmatizer.Lemmatizer()  # This is a weak Lemmatizer.


if not os.path.exists('./transformer_classification_dataset.csv'):
    X = merged_quran_vec_df_nrmlz[['original_normalized', 'lemma_normalized', 'root_normalized']].copy()
    X['شماره سوره'] = X.index.to_series().str.split('##').apply(lambda x: int(x[0]))
    X['شماره آیه'] = X.index.to_series().str.split('##').apply(lambda x: int(x[1]))

    top_sures = X.groupby('شماره سوره').count().sort_values(by='شماره آیه', ascending=False).reset_index()['شماره سوره'][:30]
    top_x = X[X['شماره سوره'].isin(top_sures)]
    top_x

    def create_data_set(out_dir: str, merged_df: pd.DataFrame, lemma_rate=0, root_rate=0, expansion_count=0):
        with open(out_dir, 'w') as fp:
            fp.write('text,labels\n')
            for col in tqdm.tqdm(['original_normalized', 'lemma_normalized', 'root_normalized']):
                for i, r in merged_df.iterrows():
                    fp.write(r[col] + ',' + str(r['شماره سوره']))
                    fp.write('\n')
            probabilities = [1 - lemma_rate - root_rate, lemma_rate, root_rate]
            functions = [lambda x: x, lemmer.lemmatize, stemmer.stem]
            for _ in range(expansion_count):  # can be enhanced
                for i, r in tqdm.tqdm(merged_df.iterrows()):
                    fp.write(
                        ' '.join([np.random.choice(a=functions, p=probabilities)(token)
                                            for token in r['original_normalized'].split()])
                    )
                    fp.write(',' + str(r['شماره سوره']))
                    fp.write('\n')

    create_data_set(out_dir='transformer_classification_dataset.csv', merged_df=top_x,
                                                                    lemma_rate=0.2,
                                                                    root_rate=0.2,
                                                                    expansion_count=5)



class ClassificationModel:

    def __init__(self, model_name, num_labels):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = None
        self.label2id = None

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                        num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.classifier = pipeline(
            "sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    @staticmethod
    def training_args_builder(output_dir="models/", learning_rate=2e-3, train_batch_size=32,
                              eval_batch_size=32, num_train_epochs=30, weight_decay=0.01):
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            save_steps=1_000
        )

    def get_train_valid_test(self, dataset, valid_ratio=0.1, test_ratio=0.1):
        train_dataset, test_valid_dataset = train_test_split(
            dataset, test_size=(valid_ratio + test_ratio), random_state=42, shuffle=True
        )
        valid_dataset, test_dataset = train_test_split(
            test_valid_dataset, test_size=((test_ratio)/(valid_ratio + test_ratio)),
            random_state=42, shuffle=True
        )

        return (
            Dataset.from_dict(train_dataset),
            Dataset.from_dict(valid_dataset),
            Dataset.from_dict(test_dataset)
        )

    def train(self, train_dataset, valid_dataset, training_args=None):
        training_args = training_args or self.training_args_builder()

        dataset = DatasetDict({"train": train_dataset, "valid": valid_dataset})

        def preprocess_function(data):
            return self.tokenizer(data["text"], truncation=True, padding=True)

        tokenized_data = dataset.map(
            preprocess_function, batched=True, remove_columns=['text'])

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        optimizer = Adafactor(
            self.model.parameters(), scale_parameter=True, 
                                    relative_step=True,
                                    warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)

        Trainer(
            eval_dataset=tokenized_data["valid"],
            train_dataset=tokenized_data["train"],
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            optimizers=(optimizer, lr_scheduler),
        ).train()

    def predict(self, test_data):
        self.classifier.model.to('cpu')
        inner_labels = self.classifier(test_data)
        return list(map(lambda lab: int(lab['label'].split('_')[1]), inner_labels))



df = pd.read_csv('./transformer_classification_dataset.csv', converters={'labels':int})
print(df.head())

model = ClassificationModel(model_name='aubmindlab/bert-base-arabertv2', num_labels=30)
train_data, valid_data, test_data = model.get_train_valid_test(df)

training_args = model.training_args_builder(
    learning_rate=1e-2,
    num_train_epochs=10,
)


model.train(train_data, valid_data, training_args)
predictions = model.predict(test_data['text'].to_list())

metric = load_metric("f1")
print("f1_score: ", metric.compute(
    predictions=predictions, references=test_data['labels'], average='micro'))

metric = load_metric("accuracy")
print("accuracy: ", metric.compute(
    predictions=predictions, references=test_data['labels']))
