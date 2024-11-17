import requests


class HuggingFaceSummarizer:
    """
    A class to generate text summaries using a Hugging Face-hosted model through an API.

    Attributes:
        api_url (str): URL for the Hugging Face inference API endpoint.
        headers (dict): Headers including the API token for authentication.
    """

    def __init__(self, api_token):
        """
        Initializes the summarizer with the Hugging Face API token.

        Args:
            api_token (str): Your Hugging Face API token for authentication.
        """
        self.api_url = "https://api-inference.huggingface.co/models/BEASTBOYJAY/my-fine-tuned-summarizer"
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def generate_summary(
        self,
        text,
        max_length=128,
        num_beams=4,
        length_penalty=1.5,
        no_repeat_ngram_size=1,
        early_stopping=True,
    ):
        """
        Generates a summary for the given input text using the Hugging Face model.

        Args:
            text (str): The input text to be summarized.
            max_length (int): Maximum length of the generated summary. Defaults to 128.
            num_beams (int): Number of beams for beam search. Defaults to 4.
            length_penalty (float): Length penalty for beam search. Defaults to 1.5.
            no_repeat_ngram_size (int): Ensures no n-gram is repeated in the summary. Defaults to 1.
            early_stopping (bool): Stops beam search early when at least one complete solution is found. Defaults to True.

        Returns:
            dict: The generated summary as a JSON object if successful, or an error message with status code.
        """
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "num_beams": num_beams,
                "length_penalty": length_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "early_stopping": early_stopping,
                "decoder_start_token_id": 101,
            },
        }

        try:
            # Make a POST request to the Hugging Face inference API
            response = requests.post(self.api_url, headers=self.headers, json=payload)

            # Raise an exception for HTTP error responses
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Return error details if the request fails
            return {
                "error": str(e),
                "status_code": response.status_code if response else "No response",
            }


if __name__ == "__main__":
    # Replace with your actual Hugging Face API token
    api_token = "Your_Access_API_Token"
    summarizer = HuggingFaceSummarizer(api_token)

    # Example input text for summarization
    text = (
        "A well-organized paragraph supports or develops a single controlling idea, "
        "which is expressed in a sentence called the topic sentence. A topic sentence "
        "has several important functions: it substantiates or supports an essay's thesis "
        "statement; it unifies the content of a paragraph and directs the order of the "
        "sentences; and it advises the reader of the subject to be discussed and how the "
        "paragraph will discuss it."
    )

    # Generate and print the summary
    summary = summarizer.generate_summary(text)
    print("Generated Summary:", summary)
