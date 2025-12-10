import whisperx
import torch
from pyannote.audio import Pipeline
import json
import os

AUDIO_FILE = "audio/meeting.mp3"
LANG = "ru"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(">>> Loading WhisperX ASR model (large-v2)...")
model = whisperx.load_model("large-v2", device)

print(">>> Running ASR...")
asr_output = model.transcribe(AUDIO_FILE, language=LANG)

print(">>> Loading alignment model...")
align_model, metadata = whisperx.load_align_model(LANG, device)

print(">>> Aligning words...")
aligned = whisperx.align(
    asr_output["segments"],
    align_model,
    metadata,
    AUDIO_FILE,
    device
)

# ---------------------------
# Pyannote Diarization
# ---------------------------

HF_TOKEN = os.getenv("HF_TOKEN")  # токен читается из окружения

if not HF_TOKEN:
    raise ValueError("Установи переменную окружения HF_TOKEN перед запуском!")

print(">>> Loading Pyannote diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)

print(">>> Running diarization...")
diarization = pipeline(AUDIO_FILE)

print(">>> Assigning speakers...")
final = whisperx.assign_word_speakers(
    diarization,
    aligned
)

# ---------------------------
# Saving results
# ---------------------------

with open("result.json", "w", encoding="utf-8") as f:
    json.dump(final["segments"], f, ensure_ascii=False, indent=2)

with open("result.txt", "w", encoding="utf-8") as f:
    for seg in final["segments"]:
        spk = seg.get("speaker", "UNK")
        text = seg.get("text", "")
        f.write(f"[{spk}] {text}\n")

print(">>> DONE! Files saved: result.json, result.txt")
