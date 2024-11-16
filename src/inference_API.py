import requests


class HuggingFaceSummarizer:
    def __init__(self, api_token):
        """
        Initialize the summarizer class with the API token.

        Parameters:
            api_token (str): Your Hugging Face API token.
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
            response = requests.post(self.api_url, headers=self.headers, json=payload)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "status_code": response.status_code if response else "No response",
            }


if __name__ == "__main__":
    api_token = "Your_Access_API_Token"
    summarizer = HuggingFaceSummarizer(api_token)

    text = (
        "A well-organized paragraph supports or develops a single controlling idea, "
        "which is expressed in a sentence called the topic sentence. A topic sentence "
        "has several important functions: it substantiates or supports an essay's thesis "
        "statement; it unifies the content of a paragraph and directs the order of the "
        "sentences; and it advises the reader of the subject to be discussed and how the "
        "paragraph will discuss it."
    )

    summary = summarizer.generate_summary(text)
    print("Generated Summary:", summary)
