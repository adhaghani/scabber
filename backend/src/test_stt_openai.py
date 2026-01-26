import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stt_module import SpeechToText

def test():
    print("Testing OpenAI Whisper (Tiny)...")
    stt = SpeechToText()
    try:
        stt.load()
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return
    
    # Generate random noise (silence/static)
    # 16kHz audio, 1 second
    audio = np.random.uniform(-0.01, 0.01, 16000).astype(np.float32)
    
    print("Transcribing noise...")
    try:
        text = stt.transcribe(audio)
        print(f"Transcription: '{text}'") # Likely empty or random words
        print("✅ Transcribe success")
    except Exception as e:
        print(f"❌ Transcribe failed: {e}")
        import traceback
        traceback.print_exc()
        
    stt.unload()

if __name__ == "__main__":
    test()
