"""
Text-to-Speech Module using Kyutai Pocket TTS
"""
from typing import Optional
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TTS_MAX_TOKENS

# Built-in Pocket TTS voices (no authentication required)
BUILTIN_VOICES = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']

# Voice cloning presets (requires HuggingFace authentication)
VOICE_PRESETS = {
    "happy": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
    "neutral": "expresso/ex01-ex01_default_006_channel1_295s.wav",
    "sad": "expresso/ex02-ex01_sad_007_channel1_334s.wav",
    "confused": "expresso/ex03-ex01_confused_005_channel1_334s.wav",
}


class TextToSpeech:
    """Text-to-Speech using Kyutai Pocket TTS (100M params, runs on CPU)"""
    
    def __init__(self, voice: str = "alba"):
        """
        Initialize TTS with Kyutai Pocket TTS
        
        Args:
            voice: Built-in voice name (alba, marius, javert, jean, fantine, cosette, eponine, azelma)
                   or path to wav file for voice cloning (requires HF auth)
        """
        self.voice_name = voice
        
        # Components
        self.tts_model = None
        self.device = None
        self.sample_rate = 24000  # Pocket TTS sample rate
        self.voice_state = None
        
    def load(self) -> None:
        """Initialize the TTS model"""
        print("🔊 Initializing Kyutai Pocket TTS...")
        
        import torch
        from pocket_tts import TTSModel
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
            print("   Using CUDA")
        else:
            self.device = "cpu"
            print("   Using CPU")
        
        # Load Pocket TTS model
        print("   Loading Pocket TTS model...")
        self.tts_model = TTSModel.load_model(
            temp=0.7,           # Gaussian temperature
            lsd_decode_steps=1  # 1-step sampling
        )
        self.tts_model = self.tts_model.to(self.device)
        
        self.sample_rate = self.tts_model.sample_rate
        
        # Load voice
        print(f"   Loading voice: {self.voice_name}...")
        

        self.voice_state = self.tts_model.get_state_for_audio_prompt(self.voice_name)

        
        print(f"   Sample rate: {self.sample_rate}")
        print("✅ Kyutai Pocket TTS initialized")
        
    def generate_audio_bytes(self, text: str) -> Optional[np.ndarray]:
        """
        Generate audio bytes from text
        
        Args:
            text: Text to convert
            
        Returns:
            numpy array of audio samples or None
        """
        if self.tts_model is None:
            raise RuntimeError("Model not initialized. Call load() first.")
            
        if not text or not text.strip():
            return None
        
        import torch
        
        # Generate audio
        with torch.no_grad():
            audio = self.tts_model.generate_audio(
                self.voice_state,
                text.strip(),
                max_tokens=TTS_MAX_TOKENS,  # Max audio tokens
                copy_state=True  # Don't modify voice state
            )
        
        # Convert to numpy
        audio_np = audio.squeeze().cpu().numpy()
        return audio_np

    def speak(self, text: str, block: bool = True) -> None:
        """
        Convert text to speech and play it
        
        Args:
            text: Text to speak
            block: Whether to block until speech is complete
        """
        audio_np = self.generate_audio_bytes(text)
        
        if audio_np is None:
            return

        import sounddevice as sd
        
        # Play audio
        if block:
            sd.play(audio_np, self.sample_rate)
            sd.wait()
        else:
            sd.play(audio_np, self.sample_rate)
            
    def speak_async(self, text: str) -> None:
        """Speak text asynchronously (non-blocking)"""
        import threading
        thread = threading.Thread(target=self.speak, args=(text, True))
        thread.daemon = True
        thread.start()
        
    def stop(self) -> None:
        """Stop current speech"""
        import sounddevice as sd
        sd.stop()
        
    def save_to_file(self, text: str, filename: str) -> None:
        """
        Save speech to audio file
        
        Args:
            text: Text to convert
            filename: Output filename (e.g., 'output.wav')
        """
        if self.tts_model is None:
            raise RuntimeError("Model not initialized. Call load() first.")
            
        import torch
        import wave
        
        # Generate audio
        with torch.no_grad():
            audio = self.tts_model.generate_audio(
                self.voice_state,
                text.strip(),
                max_tokens=TTS_MAX_TOKENS,
                copy_state=True
            )
        
        audio_np = audio.squeeze().cpu().numpy()
        
        # Save as WAV file
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
            
        print(f"✅ Saved speech to: {filename}")
        
    def list_voices(self) -> None:
        """List available voice presets"""
        print("\n🎤 Built-in Voices (no auth required):")
        print("-" * 40)
        for name in BUILTIN_VOICES:
            marker = " (current)" if name == self.voice_name else ""
            print(f"  • {name}{marker}")
        print()
        print("🎤 Voice Cloning Presets (requires HF auth):")
        print("-" * 40)
        for name in VOICE_PRESETS.keys():
            marker = " (current)" if name == self.voice_name else ""
            print(f"  • {name}{marker}")
        print("-" * 40)
        print("  Or provide path to any .wav file")
        
    def set_voice(self, voice_name: str) -> None:
        """Switch to a different voice (requires reload)"""
        if voice_name in BUILTIN_VOICES or voice_name in VOICE_PRESETS or voice_name.endswith('.wav'):
            self.voice_name = voice_name
            print(f"✅ Voice set to: {voice_name}")
            print("   Note: Call load() again to use the new voice")
        else:
            print(f"❌ Unknown voice: {voice_name}")
            print(f"   Built-in voices: {BUILTIN_VOICES}")
            print(f"   Cloning presets: {list(VOICE_PRESETS.keys())}")
            print("   Or provide a path to a .wav file")
        
    def unload(self) -> None:
        """Clean up TTS model"""
        import gc
        import torch
        
        if self.tts_model:
            del self.tts_model
            self.tts_model = None
        if self.voice_state is not None:
            del self.voice_state
            self.voice_state = None
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("🗑️  TTS model unloaded")


def test_tts():
    """Test TTS module"""
    print("\n🔊 Kyutai Pocket TTS Test")
    print("=" * 40)
    
    tts = TextToSpeech(voice="alba")  # Use built-in voice
    tts.load()
    
    # List available voices
    tts.list_voices()
    
    # Test speech
    test_texts = [
        "Hello! I am your voice assistant.",
        "How can I help you today?",
        "This is a test of the Kyutai Pocket text to speech system."
    ]
    
    for text in test_texts:
        print(f"\n📢 Speaking: {text}")
        tts.speak(text)
    
    # Cleanup
    tts.unload()
    print("\n✅ TTS test complete!")


if __name__ == "__main__":
    test_tts()
