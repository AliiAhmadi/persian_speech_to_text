from pydub import AudioSegment
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from fuzzywuzzy import fuzz
import os

def split_audio_overlap(input_file, chunk_length_ms=30000, overlap_ms=5000):
    audio = AudioSegment.from_file(input_file)
    step = chunk_length_ms - overlap_ms
    chunks = []
    os.makedirs("chunks_overlap", exist_ok=True)

    for start in range(0, len(audio), step):
        end = min(start + chunk_length_ms, len(audio))
        chunk = audio[start:end]
        path = f"chunks_overlap/chunk_{start//1000}_{end//1000}.wav"
        chunk.export(path, format="wav")
        chunks.append(path)
    
    return chunks

def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy()

def transcribe(audio_path, processor, model):
    audio_input = load_audio(audio_path)
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")

    generated_ids = model.generate(
        inputs,
        forced_decoder_ids=processor.get_decoder_prompt_ids(language="fa", task="transcribe"),
        num_beams=1
    )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def merge_transcripts_fuzzy(texts, window=15, threshold=85):
    merged = texts[0]
    for i in range(1, len(texts)):
        last_part = merged[-window:]
        current = texts[i]

        for j in range(window):
            overlap_candidate = current[:j+1]
            score = fuzz.ratio(last_part[-(j+1):], overlap_candidate)

            if score > threshold:
                current = current[j+1:].strip()
                break

        merged += " " + current

    return merged.strip()

if __name__ == "__main__":
    processor = WhisperProcessor.from_pretrained("AliiAhmadi/whisper-fa")
    model = WhisperForConditionalGeneration.from_pretrained("AliiAhmadi/whisper-fa").to("cuda")


    # Your audio file:
    input_file = "x.mp3"

    audio_chunks = split_audio_overlap(input_file)
    transcripts = []

    for path in audio_chunks:
        text = transcribe(path, processor, model)
        transcripts.append(text)
        print(f"Transcribed {os.path.basename(path)}: {text}")

    final_text = merge_transcripts_fuzzy(transcripts)
    print("\nFinal Transcription:\n", final_text)
