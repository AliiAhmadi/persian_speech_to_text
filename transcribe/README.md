# Audio Transcription with Whisper Model & Fuzzy Matching

This project splits a long audio file into smaller overlapping chunks, processes them using the Whisper model for transcription, and merges the results with fuzzy matching to remove duplicated phrases. This technique is useful for transcribing long audio files with high accuracy and avoiding repeated words or phrases at the boundaries of chunks.

## Requirements

Before running the code, ensure you have the following dependencies installed:

```bash
pip install torch torchaudio transformers pydub fuzzywuzzy python-Levenshtein
```

### How the Code Works

1. **Audio Splitting:**
   - The audio file is split into 30-second chunks with 5-second overlap.
   - This overlap helps prevent missing out on important words or phrases at the boundaries of chunks.

2. **Transcription:**
   - Each audio chunk is passed through the Whisper model to transcribe it into text.
   - The Whisper model used here is `steja/whisper-large-persian`, a version fine-tuned for Persian language.

3. **Fuzzy Matching:**
   - After transcription, overlapping parts of the text are compared using fuzzy matching.
   - The fuzzy matching algorithm checks if parts of the ending of the previous chunk are repeated at the beginning of the next chunk.
   - If the overlap exceeds a certain similarity threshold (85% by default), the repeated part is removed.

4. **Final Output:**
   - The transcribed text chunks are merged into one long, clean transcription.
   - The result is printed as the final transcription.

---

## Explanation of Key Concepts

### 1. **Audio Splitting and Overlap:**

The `split_audio_overlap` function splits the audio into smaller chunks to process them individually. The `chunk_length_ms` parameter controls the length of each chunk (default: 30 seconds), and the `overlap_ms` parameter controls how much the chunks overlap (default: 5 seconds). The overlap ensures that no part of the audio is missed, especially when speech is continuous across chunk boundaries.

### 2. **Whisper Model:**

We use the `WhisperProcessor` and `WhisperForConditionalGeneration` classes from Hugging Face to load and run the Whisper model for transcription. The model is specifically trained for Persian audio, so it is suited for transcribing Persian speech.

### 3. **Fuzzy Matching for Text Merging:**

Fuzzy matching is used to compare the last part of a chunk with the first part of the next chunk and remove any duplicate text. The `fuzzywuzzy` library is used to calculate the similarity score between two text fragments.

The `merge_transcripts_fuzzy` function performs the following:
- It takes in the list of transcriptions from the chunks.
- For each chunk, it compares the ending part of the previous transcription with the beginning of the current one.
- If the similarity score exceeds a threshold (85% by default), the overlapping part is removed from the current chunk.

#### Example of Fuzzy Matching:

For example, if we have the following two chunks:

- **Chunk 1**: "سلام دوستان عزیز خوش آمدید به برنامه ما"
- **Chunk 2**: "دوستان عزیز خوش آمدید به برنامه ما"

Without fuzzy matching, the combined result would be:

```
سلام دوستان عزیز خوش آمدید به برنامه ما دوستان عزیز خوش آمدید به برنامه ما
```

But with fuzzy matching, the overlapping "دوستان عزیز خوش آمدید به برنامه ما" part is removed, resulting in:

```
سلام دوستان عزیز خوش آمدید به برنامه ما
```

### 4. **Tuning Fuzzy Matching:**

- `threshold`: The similarity threshold for fuzzy matching (default is 85%). If the similarity between the end of the previous chunk and the start of the next chunk is above this threshold, the overlap is removed.
- `window`: The number of characters to compare when checking for overlap (default is 15). This defines the length of the overlap portion to check.

---

## Conclusion

This approach ensures accurate transcription of long audio files while avoiding repetitive parts at chunk boundaries. By leveraging fuzzy matching, the final transcription is cleaner and more coherent.

---

## License

This project is licensed under the MIT License.
