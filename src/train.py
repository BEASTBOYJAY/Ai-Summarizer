from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset


class TrainBERTSummarizer:
    """
    A class to train a BERT-based Encoder-Decoder model for text summarization.

    Attributes:
        tokenizer (BertTokenizer): Tokenizer for preprocessing text.
        model (EncoderDecoderModel): Encoder-Decoder model for summarization.
        dataset (Dataset): Loaded dataset for training.
        tokenized_dataset (Dataset): Preprocessed dataset for training.
        data_collator (DataCollatorForSeq2Seq): Data collator for padding and batching.
        trainer (Trainer): Trainer for model training.
    """

    def __init__(
        self, encoder_model="bert-base-uncased", decoder_model="bert-base-uncased"
    ):
        """
        Initializes the TrainBERTSummarizer class with the specified encoder and decoder models.

        Args:
            encoder_model (str): Name or path of the pretrained encoder model. Defaults to 'bert-base-uncased'.
            decoder_model (str): Name or path of the pretrained decoder model. Defaults to 'bert-base-uncased'.
        """
        self.tokenizer = BertTokenizer.from_pretrained(encoder_model)
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_model, decoder_model
        )
        self._set_model_config()

    def _set_model_config(self):
        """
        Configures model-specific settings for summarization tasks, such as token IDs, max length,
        and beam search parameters.
        """
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
        """
        Loads a dataset for training.

        Args:
            dataset_name (str): Name of the dataset to load. Defaults to 'cnn_dailymail'.
            dataset_version (str): Version of the dataset. Defaults to '3.0.0'.
            split (str): Subset of the dataset to load (e.g., 'train[:1%]'). Defaults to 'train[:1%]'.
        """
        self.dataset = load_dataset(dataset_name, dataset_version, split=split)
        print(f"Loaded {split} of {dataset_name} dataset.")

    def preprocess_data(self):
        """
        Preprocesses the dataset by tokenizing the input and target texts and preparing them
        for model training.
        """

        def preprocess_function(examples):
            """
            Tokenizes input articles and summaries.

            Args:
                examples (dict): A batch of examples containing 'article' and 'highlights'.

            Returns:
                dict: Tokenized inputs with attention masks and labels.
            """
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
        """
        Trains the model on the preprocessed dataset.

        Args:
            output_dir (str): Directory to save training outputs. Defaults to './results'.
            learning_rate (float): Learning rate for the optimizer. Defaults to 2e-5.
            batch_size (int): Batch size for training and evaluation. Defaults to 2.
            num_epochs (int): Number of training epochs. Defaults to 3.
            weight_decay (float): Weight decay for regularization. Defaults to 0.01.
        """
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

        self.trainer.train()
        print("Training completed.")


if __name__ == "__main__":
    summarizer = TrainBERTSummarizer()
    summarizer.load_data()
    summarizer.preprocess_data()
    summarizer.train()
