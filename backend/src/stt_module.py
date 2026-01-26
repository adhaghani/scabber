""" Speech-to-Text Module using OpenAI Whisper (Offline) """
import numpy as np
import whisper
import os
import sys
from typing import Optional, List, Tuple, Generator

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import WHISPER_MODEL_ID, STT_SAMPLE_RATE, MODELS_DIR, DEVICE

class SpeechToText:
    """Speech-to-Text using OpenAI Whisper"""
    
    def __init__(
        self,
        model_name: str = WHISPER_MODEL_ID,
        device: str = DEVICE
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        
    def load(self) -> None:
        """Load the Whisper model"""
        print(f"📝 Loading OpenAI Whisper model: {self.model_name}")
        print(f"   Device: {self.device}")
        
        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        try:
            # check if device is mps, openai-whisper supports mps in recent versions
            # but sometimes cpu is safer if mps isn't fully supported.
            # Using whatever is passed in config, falling back if needed.
            
            # openai-whisper download_root defaults to ~/.cache/whisper
            # we can set it to MODELS_DIR if we want, but default is fine for global cache.
            
            self.model = whisper.load_model(self.model_name, device=self.device)
            print("✅ OpenAI Whisper model loaded")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en"
    ) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio: numpy array of audio samples
            language: Language code
            
        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if len(audio) == 0:
            return ""
        
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            
        # Transcribe
        # openai-whisper supports numpy array directly
        # language="english" improves accuracy if we know it's English
        result = self.model.transcribe(audio, language="english", fp16=False) # fp16=False often safer on CPU/MPS mixed
        
        text = result["text"].strip()
        return text
    
    def transcribe_stream(
        self,
        audio: np.ndarray,
        language: str = "en"
    ) -> Generator[str, None, None]:
        """
        Transcribe audio in chunks (simulated streaming)
        """
        if self.model is None:
             raise RuntimeError("Model not loaded. Call load() first.")

        text = self.transcribe(audio, language)
        if text:
            yield text

    def transcribe_with_timestamps(
        self,
        audio: np.ndarray,
        language: str = "en"
    ) -> List[Tuple[float, float, str]]:
        """
        Transcribe audio with timestamps
        """
        if self.model is None:
             raise RuntimeError("Model not loaded. Call load() first.")

        # transcribe returns segments with timestamps
        result = self.model.transcribe(audio, language="english", fp16=False)
        
        segments = result.get("segments", [])
        return [(s["start"], s["end"], s["text"]) for s in segments]
    
    def unload(self) -> None:
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except:
            pass
        
        print("🗑️ Whisper model unloaded")


def test_stt():
    """Test STT module with a sample recording"""
    print("Run this module directly to test: python -m src.stt_module")
    pass

if __name__ == "__main__":
    from audio_utils import AudioRecorder
    
    print("\n🎙️ Speech-to-Text Test (OpenAI Whisper)")
    print("=" * 40)
    
    # Initialize
    stt = SpeechToText()
    stt.load()
    recorder = AudioRecorder()
    
    # Record
    print("\nSpeak something (5 seconds)...")
    audio = recorder.record_fixed_duration(duration=5)
    
    # Transcribe
    print("\n🔄 Transcribing...")
    text = stt.transcribe(audio)
    print(f"\n📝 Final Transcription: {text}")
    print("\n✅ STT test complete!")