from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def train_biobert(labeled_data, model_name="dmis-lab/biobert-base-cased-v1.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    dataset = Dataset.from_list(labeled_data)
    tokenized_dataset = dataset.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir="./biobert",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
        evaluation_strategy="no"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()
    return model
