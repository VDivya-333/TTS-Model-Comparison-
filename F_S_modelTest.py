import os
import numpy as np
import pandas as pd
import jiwer
import soundfile as sf
from transformers import pipeline, BarkModel, AutoProcessor

# =======================
# 1️⃣ Folder Setup
# =======================
os.makedirs("suno_outputaudio", exist_ok=True)
os.makedirs("facebook_outputaudio", exist_ok=True)

# =======================
# 2️⃣ Load Input Data
# =======================
df = pd.read_csv("input.csv")  

# =======================
# 3️⃣ Load TTS Models
# =======================
print("Loading models...")
SUNO_MODEL = "suno/bark-small"
MMS_MODEL = "facebook/mms-tts-eng"
ASR_MODEL = "openai/whisper-tiny"

print(f"🔹 Loading Suno Bark TTS: {SUNO_MODEL}")
# We load the model and processor manually to use voice presets for cleaner audio
suno_model = BarkModel.from_pretrained(SUNO_MODEL)
suno_processor = AutoProcessor.from_pretrained(SUNO_MODEL)

# Use a voice preset for cleaner output (reduces background noise)
voice_preset = "v2/en_speaker_6"

print(f"🔹 Loading Facebook MMS TTS: {MMS_MODEL}")
fac_tts = pipeline("text-to-speech", model=MMS_MODEL)

# =======================
# 4️⃣ Load ASR Model
# =======================
asr = pipeline("automatic-speech-recognition", model=ASR_MODEL)
print("✅ Models loaded successfully.\n")

asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


def evaluate_tts(tts_pipeline, text, out_path, is_suno=False):
    """
    Generates speech, saves audio, transcribes back, computes WER-based accuracy.
    """
    # Generate TTS
    if is_suno:
        # Use the processor and model directly for Suno to apply the voice preset
        inputs = suno_processor(text, voice_preset=voice_preset, return_tensors="pt")
        speech_values = suno_model.generate(**inputs, do_sample=True)
        audio = speech_values.cpu().numpy().squeeze()
        sr = suno_model.generation_config.sample_rate
    else:
        output = tts_pipeline(text)
        audio = output["audio"]
        sr = output["sampling_rate"]

    # 🔹 Ensure audio is numpy float32
    if hasattr(audio, "numpy"):  # torch tensor
        audio = audio.numpy()
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    audio = audio.astype(np.float32)

    # 🔹 Squeeze 2D audio (e.g., from Suno) to 1D for saving
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio.squeeze()

    # Save audio
    sf.write(out_path, audio, sr)

    # Run ASR
    transcription = asr({"array": audio, "sampling_rate": sr})["text"]

    # Compute accuracy
    wer = jiwer.wer(text.lower(), transcription.lower())
    accuracy = round((1 - wer) * 100, 2)
    return accuracy


# =======================
# 5️⃣ Evaluate All Sentences
# =======================
results = []

for idx, row in df.iterrows():
    text = row["text"]
    sentence_id = row["id"]

    print(f"\n🔹 Sentence {sentence_id}: {text}")

    # Suno Bark
    # We pass the pipeline components and a flag to handle it differently
    suno_path = f"suno_outputaudio/suno_audio{sentence_id}.wav"
    suno_acc = evaluate_tts(None, text, suno_path, is_suno=True)

    # Facebook MMS
    fac_path = f"facebook_outputaudio/facebook_audio{sentence_id}.wav"
    fac_acc = evaluate_tts(fac_tts, text, fac_path)

    print(f"✅ Suno Accuracy: {suno_acc}% | ✅ Facebook Accuracy: {fac_acc}%")

    # Append results
    results.append([sentence_id, text, fac_acc, suno_acc])

# =======================
# 6️⃣ Save Final Results CSV
# =======================
out_df = pd.DataFrame(results, columns=["id", "text", "fac_accuracy", "suno_accuracy"])
out_df.to_csv("tts_results.csv", index=False)

print("\n🎉 Completed! Results saved in tts_results.csv and audio folders.")
