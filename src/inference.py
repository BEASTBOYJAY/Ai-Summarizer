from transformers import EncoderDecoderModel, BertTokenizer


class TextSummarizer:
    """
    A class for generating text summaries using a fine-tuned Encoder-Decoder model.

    Attributes:
        tokenizer (BertTokenizer): Tokenizer for preprocessing input text.
        model (EncoderDecoderModel): Fine-tuned Encoder-Decoder model for summarization.
    """

    def __init__(self, model_path, tokenizer_name="bert-base-uncased"):
        """
        Initializes the TextSummarizer with a pretrained model and tokenizer.

        Args:
            model_path (str): Path to the pretrained model.
            tokenizer_name (str): Name or path of the tokenizer. Defaults to 'bert-base-uncased'.
        """
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = EncoderDecoderModel.from_pretrained(model_path)

    def summarize(self, text, max_input_length=512):
        """
        Generates a summary for the given input text.

        Args:
            text (str): Input text to be summarized.
            max_input_length (int): Maximum length of the input text for tokenization. Defaults to 512.

        Returns:
            str: The generated summary.
        """
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_input_length,
        )

        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_start_token_id=self.tokenizer.cls_token_id,
            max_length=128,
            num_beams=4,
            length_penalty=1.5,
            no_repeat_ngram_size=1,
            early_stopping=True,
        )

        # Decode and return the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


if __name__ == "__main__":
    # Initialize the summarizer with a fine-tuned model
    summarizer = TextSummarizer(model_path="BEASTBOYJAY/my-fine-tuned-summarizer")

    # Example article for summarization
    test_article = (
        "SCIENTISTS HAVE LEARNED TO SUPPLEMENT THE SENSE OF SIGHT IN NUMEROUS WAYS. "
        "In front of the tiny pupil of the eye they put, on Mount Palomar, a great monocle 200 inches in diameter, "
        "and with it see 2000 times farther into the depths of space. Or they look through a small pair of lenses "
        "arranged as a microscope into a drop of water or blood, and magnify by as much as 2000 diameters the living "
        "creatures there, many of which are among mans most dangerous enemies. Or, if we want to see distant happenings "
        "on earth, they use some of the previously wasted electromagnetic waves to carry television images which they "
        "re-create as light by whipping tiny crystals on a screen with electrons in a vacuum. Or they can bring happenings "
        "of long ago and far away as colored motion pictures, by arranging silver atoms and color-absorbing molecules to "
        "force light waves into the patterns of original reality. Or if we want to see into the center of a steel casting "
        "or the chest of an injured child, they send the information on a beam of penetrating short-wave X rays, and then "
        "convert it back into images we can see on a screen or photograph. THUS ALMOST EVERY TYPE OF ELECTROMAGNETIC RADIATION "
        "YET DISCOVERED HAS BEEN USED TO EXTEND OUR SENSE OF SIGHT IN SOME WAY."
    )

    # Generate and print the summary
    summary = summarizer.summarize(test_article)
    print("Generated Summary:", summary)
