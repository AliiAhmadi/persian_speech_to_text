from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="AliiAhmadi/whisper-fa",
    device="cuda"
)

result = pipe(
    "x.mp3",
    chunk_length_s=30,
    stride_length_s=5,
    return_timestamps=False
)

print(result["text"])
