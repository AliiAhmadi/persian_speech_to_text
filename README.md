# Fine-Tuning OpenAI Whisper for Persian Speech Recognition

A complete and practical guide for **fine-tuning OpenAI's Whisper model** on **Persian language datasets** to achieve **high-accuracy automatic speech recognition (ASR)** for Farsi audio.


## Project Overview

This repository provides a clear, reproducible pipeline for **fine-tuning Whisper** — OpenAI's powerful speech-to-text model — using **Persian speech datasets**.  
The goal is to enhance Whisper’s performance specifically for **Farsi transcriptions** by training it on native Persian audio and text.


## Key Features

- Fine-tunes Whisper on **Persian speech** for improved ASR results  
- Step-by-step code using **Hugging Face Transformers**  
- Uses **WER (Word Error Rate)** for accurate evaluation  
- Supports **Common Voice Persian dataset** (or your own dataset)  
- Designed for easy customization and GPU-accelerated training  


## Why Fine-Tune Whisper for Persian?

Although OpenAI's Whisper supports multiple languages, fine-tuning it on **Persian-specific datasets** can drastically improve transcription quality.  
This is especially useful for:
- Voice assistant applications in Farsi  
- Persian audio-to-text services  
- Academic and commercial speech AI research  


## Requirements

- Python 3.8+
- PyTorch with GPU support
- Hugging Face `transformers` and `datasets`
- `evaluate` for metric calculation (Word Error Rate)
