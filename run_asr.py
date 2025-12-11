import os
import json
import torch
import whisperx
from pyannote.audio import Pipeline

# -----------------------------
# SETTINGS
# -----------------------------
AUDIO_FILE = "audio/meeting.mp3"   # путь к файлу
LANG = "ru"

# -----------------------------
# CHECK CUDA
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f">>> Device selected: {device}")

# -----------------------------
# CHECK HF TOKEN
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("ERROR: Переменная HF_TOKEN не установлена.\n"
                     "Сделай export HF_TOKEN=\"hf_xxxxx\" перед запуском.")

# -----------------------------
# 1. LOAD WHISPERX MODEL
# -----------------------------
print(">>> Loading WhisperX ASR model (large-v2)...")
model = whisperx.load_model("large-v2", device)

# -----------------------------
# 2. RUN STT
# -----------------------------
print(">>> Running ASR...")
asr_out = model.transcribe(AUDIO_FILE, language=LANG)

# -----------------------------
# 3. LOAD ALIGNMENT MODEL
# -----------------------------
print(">>> Loading alignment model...")
align_model, metadata = whisperx.load_align_model(LANG, device)

print(">>> Aligning words...")
aligned = whisperx.align(
    asr_out["segments"],
    align_model,
    metadata,
    AUDIO_FILE,
    device
)

# -----------------------------
# 4. DIARIZATION
# -----------------------------
print(">>> Loading Pyannote diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)

print(">>> Running diarization...")
diarization = pipeline(AUDIO_FILE)

# -----------------------------
# 5. ASSIGN SPEAKERS
# -----------------------------
print(">>> Assigning speakers to segments...")
final = whisperx.assign_word_speakers(
    diarization,
    aligned
)

# -----------------------------
# 6. SAVE RESULTS
# -----------------------------
print(">>> Saving result.json ...")
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(final["segments"], f, ensure_ascii=False, indent=2)

print(">>> Saving result.txt ...")
with open("result.txt", "w", encoding="utf-8") as f:
    for seg in final["segments"]:
        speaker = seg.get("speaker", "SPK")
        text = seg.get("text", "")
        f.write(f"[{speaker}] {text}\n")

print("\n===============================")
print(">>> DONE! Files saved:")
print("result.json")
print("result.txt")
print("===============================")
