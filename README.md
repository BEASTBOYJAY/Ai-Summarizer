
# Text Summarization Pipeline

This repository provides tools to train and evaluate a custom text summarization model. It includes:

- **Training Script** (`train.py`): Fine-tunes a BERT-based Encoder-Decoder model for summarization tasks.
- **Inference Script** (`inference.py`): Generates summaries using the fine-tuned model.
- **API Script** (`inference_API.py`): Enables web-based summarization via a Hugging Face-hosted API which is being used in the deployed Webapp.

## Features

- **Custom Training**: Fine-tune a summarization model on any dataset.
- **Inference Locally**: Use the trained model for local summarization.
- **API Deployment**: Leverage Hugging Face API for web app integration.



## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/BEASTBOYJAY/Ai-Summarizer.git
   cd Ai-Summarizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

1. Prepare your dataset.
   - Default: `train.py` uses the `cnn_dailymail` dataset.
   - Modify `train.py` to include your dataset if needed.

2. Train the model:
   ```bash
   python train.py
   ```
   Output: A trained model saved in the `./results` directory.

---

### Local Inference

1. Ensure the trained model is saved in a directory.
2. Use the `inference.py` script:
   ```bash
   python inference.py
   ```

   Replace `model_path` with the directory containing your fine-tuned model.



### API Deployment

1. Set up your Hugging Face model and API token:
   - Upload your fine-tuned model to Hugging Face.
   - Add your API token in `inference_API.py`.

2. Run the API script:
   ```bash
   python inference_API.py
   ```


## Examples

### Training
```bash
python train.py
```

### Local Inference
```python
from inference import TextSummarizer

summarizer = TextSummarizer(model_path="./results/last-checkpoint")
summary = summarizer.summarize("Your text here.")
print(summary)
```

