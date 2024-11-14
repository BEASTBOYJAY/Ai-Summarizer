from transformers import EncoderDecoderModel, BertTokenizer


class TextSummarizer:
    def __init__(
        self, model_path="./results/checkpoint-4308", tokenizer_name="bert-base-uncased"
    ):
        # Initialize the tokenizer and load the fine-tuned model
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = EncoderDecoderModel.from_pretrained(model_path)
        self._set_generation_config()

    def _set_generation_config(self):
        # Configure generation parameters
        self.model.config.decoder_start_token_id = (
            self.tokenizer.cls_token_id
        )  # Start token
        self.model.config.bos_token_id = (
            self.tokenizer.cls_token_id
        )  # Beginning-of-sequence token
        self.model.config.eos_token_id = (
            self.tokenizer.sep_token_id
        )  # End-of-sequence token
        self.model.config.max_length = 128
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4

    def summarize(self, text, max_input_length=512):
        # Tokenize the input text and generate the summary
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_input_length,
        )
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=self.model.config.max_length,
            num_beams=self.model.config.num_beams,
            length_penalty=self.model.config.length_penalty,
            early_stopping=True,
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


# Example usage
if __name__ == "__main__":
    summarizer = TextSummarizer()
    test_article = (
        "The BERT Model revolutionized NLP and with its easily fine-tuned parameters to different NLP tasks. "
        "In particular, the task of text summarization has been researched intensively in the subfields of "
        "abstractive and extractive text summarization. The goal of extractive text summarization models is to "
        "score each sentence in the document to be able to include the most relevant sentences in the summary. "
        "In the case of abstractive summarization, there is a need for the model to have word generative "
        "capabilities given words or context that might not be included in the document. The progress in the "
        "extractive text summarization has seen remarkable accuracy thanks to models like BERTSUM which uses "
        "fine-tuning layers to add document-based context from the BERT outputs to more efficient models such "
        "as DistilBERT, which shows relatively similar performance but requires less space and time to run."
    )
    summary = summarizer.summarize(test_article)
    print("Generated Summary:", summary)
