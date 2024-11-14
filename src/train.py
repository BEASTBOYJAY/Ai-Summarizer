from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset


class TrainBERTSummarizer:
    def __init__(
        self, encoder_model="bert-base-uncased", decoder_model="bert-base-uncased"
    ):
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(encoder_model)
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_model, decoder_model
        )
        self._set_model_config()

    def _set_model_config(self):
        # Configure the model for summarization
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.max_length = 128
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4

    def load_data(
        self, dataset_name="cnn_dailymail", dataset_version="3.0.0", split="train[:1%]"
    ):
        # Load the dataset
        self.dataset = load_dataset(dataset_name, dataset_version, split=split)
        print(f"Loaded {split} of {dataset_name} dataset.")

    def preprocess_data(self):
        # Preprocess the dataset
        def preprocess_function(examples):
            inputs = self.tokenizer(
                examples["article"],
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            labels = self.tokenizer(
                examples["highlights"],
                max_length=128,
                truncation=True,
                padding="max_length",
            )
            inputs["labels"] = labels["input_ids"]
            inputs["attention_mask"] = inputs["attention_mask"]
            return inputs

        self.tokenized_dataset = self.dataset.map(preprocess_function, batched=True)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

    def train(
        self,
        output_dir="./results",
        learning_rate=2e-5,
        batch_size=2,
        num_epochs=3,
        weight_decay=0.01,
    ):
        # Set up training arguments and Trainer
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            eval_dataset=self.tokenized_dataset,
            data_collator=self.data_collator,
        )

        # Start training
        self.trainer.train()
        print("Training completed.")


# Example usage
if __name__ == "__main__":
    summarizer = TrainBERTSummarizer()
    summarizer.load_data()
    summarizer.preprocess_data()
    summarizer.train()
