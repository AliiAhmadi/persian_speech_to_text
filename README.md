
# Whisper Fine-Tuning Project

This project focuses on fine-tuning the [Whisper](https://github.com/openai/whisper) model for automatic speech recognition (ASR). The goal is to enhance the performance of Whisper on a custom dataset by using transfer learning and optimizing model parameters. The project leverages the Hugging Face `transformers` library for training and evaluation.

## Features

- Fine-tuning Whisper for speech-to-text tasks on custom audio data.
- Supports mixed precision training (`fp16`) for faster training on GPUs.
- Integration with [Weights & Biases](https://wandb.ai/) for experiment tracking and monitoring.
- Tokenization of both audio and text inputs for model training.
- Evaluation during each epoch to track performance improvements.
- Model checkpoint saving to allow resumption of training.

## Requirements

- Python 3.7 or higher
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [PyTorch](https://pytorch.org/)
- [Weights & Biases](https://wandb.ai/) (for experiment tracking)
- Additional libraries: `datasets`, `tqdm`, `torchaudio`

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/your_username/whisper-finetuning.git
cd whisper-finetuning
```

### Step 2: Install dependencies

It is recommended to use a virtual environment. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Step 3: Setup Weights & Biases (optional)

For experiment tracking, create a W&B account and set up the API key:

```bash
wandb login
```

## Usage

### Step 1: Prepare the Dataset

Ensure you have a dataset for fine-tuning. You can use any ASR dataset in the correct format (e.g., audio files with transcriptions). The dataset should have fields such as `audio` and `transcript`.

### Step 2: Tokenize the Dataset

This project uses a custom tokenization function to preprocess the dataset:

```python
def encode_audio(examples):
    audio_input = tokenizer(examples["audio"], return_tensors="pt", padding=True, truncation=True)
    return audio_input

train_dataset = train_dataset.map(encode_audio, batched=True)
```

### Step 3: Fine-Tune the Model

Run the training script to start fine-tuning the Whisper model:

```bash
python train.py
```

### Step 4: Evaluate the Model

Evaluation occurs at the end of each epoch, and model checkpoints are saved automatically.

```bash
python evaluate.py
```

### Step 5: Model Inference

After training, use the fine-tuned model to make predictions on new audio data:

```python
from transformers import pipeline

model = WhisperForConditionalGeneration.from_pretrained("./whisper_fine_tuned")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large")

transcriber = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer)

result = transcriber("path_to_audio_file.wav")
print(result["text"])
```

## Project Structure

```
whisper-finetuning/
├── data/                    # Dataset (audio and transcriptions)
├── logs/                    # Logs for experiment tracking (via Weights & Biases)
├── model/                   # Fine-tuned model checkpoints
├── train.py                 # Script to start model training
├── evaluate.py              # Script for evaluating the model
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── ...                      # Other helper files and scripts
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Whisper Model by OpenAI](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [Weights & Biases](https://wandb.ai/)
- [PyTorch](https://pytorch.org/)
