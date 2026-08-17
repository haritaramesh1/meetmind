from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
MEETINGS = ROOT / "meetings"
OUTPUTS = ROOT / "outputs"
MEETINGS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN is missing; speaker diarization will fail.")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is missing; minutes generation will fail.")
