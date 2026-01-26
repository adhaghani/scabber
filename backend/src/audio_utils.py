"""
Audio recording utilities for Voice AI Assistant
With real-time Voice Activity Detection (VAD) using energy-based detection
"""
import sounddevice as sd
import numpy as np
import queue
import threading
from typing import Optional, Callable, Generator
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SAMPLE_RATE, CHANNELS, RECORDING_DURATION, SILENCE_THRESHOLD, SILENCE_DURATION


class VADDetector:
    """
    Lightweight Voice Activity Detector using energy-based detection
    No torch/torchaudio dependencies - uses simple RMS energy threshold
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.01,  # Lower threshold for better sensitivity
        min_speech_duration_ms: int = 250,
        silence_duration_ms: int = 800,
        padding_duration_ms: int = 200  # Add padding before/after speech
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.silence_duration_ms = silence_duration_ms
        self.padding_duration_ms = padding_duration_ms
        
        # State
        self.triggered = False
        self.temp_end = 0
        self.current_speech_ms = 0
        
        # Audio buffer for the current utterance
        self.buffer = []
        
        # Pre-trigger buffer for padding
        self.pre_buffer = []
        self.pre_buffer_size = int(padding_duration_ms * sample_rate / 1000)
        
    def _compute_energy(self, chunk: np.ndarray) -> float:
        """Compute RMS energy of audio chunk"""
        if len(chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(chunk ** 2)))
            
    def process_chunk(self, chunk: np.ndarray) -> dict:
        """
        Process an audio chunk and return state
        
        Args:
            chunk: Float32 numpy array of audio samples
            
        Returns:
            dict with keys:
                'is_speech': bool (current chunk has speech)
                'state': 'silence', 'start', 'speech', 'end'
                'speech_audio': np.ndarray (if 'end', returns the full utterance)
        """
        chunk_ms = len(chunk) * 1000 / self.sample_rate
        
        # Compute energy for VAD
        energy = self._compute_energy(chunk)
        is_speech = energy > self.threshold
        
        result = {
            'is_speech': is_speech,
            'state': 'silence',
            'speech_audio': None,
            'energy': energy  # For debugging
        }
        
        if is_speech:
            # Speech active
            if not self.triggered:
                self.triggered = True
                result['state'] = 'start'
                self.current_speech_ms = 0
                
                # Add pre-buffer (context before speech started)
                if self.pre_buffer:
                    self.buffer.extend(self.pre_buffer)
                    self.pre_buffer = []
            else:
                result['state'] = 'speech'
            
            self.current_speech_ms += chunk_ms
            self.temp_end = 0
            self.buffer.append(chunk)
            
        elif self.triggered:
            # Silence during speech (potentially end)
            self.temp_end += chunk_ms
            self.buffer.append(chunk)
            
            if self.temp_end >= self.silence_duration_ms:
                # End of speech detected
                if self.current_speech_ms >= self.min_speech_duration_ms:
                    result['state'] = 'end'
                    result['speech_audio'] = np.concatenate(self.buffer)
                else:
                    # Too short, discard
                    result['state'] = 'silence'
                
                # Reset state
                self.triggered = False
                self.temp_end = 0
                self.current_speech_ms = 0
                self.buffer = []
                self.pre_buffer = []
            else:
                result['state'] = 'speech'  # Still in speech, just quiet
        else:
            # Silence - maintain pre-buffer for context
            self.pre_buffer.append(chunk)
            # Keep only recent audio in pre-buffer
            total_samples = sum(len(c) for c in self.pre_buffer)
            while total_samples > self.pre_buffer_size and len(self.pre_buffer) > 1:
                removed = self.pre_buffer.pop(0)
                total_samples -= len(removed)
            
        return result

    def reset(self):
        """Reset VAD state"""
        self.triggered = False
        self.temp_end = 0
        self.current_speech_ms = 0
        self.buffer = []
        self.pre_buffer = []


class AudioRecorder:
    """Handles audio recording from microphone with real-time VAD"""
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        default_duration: float = RECORDING_DURATION
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.default_duration = default_duration
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # VAD components
        self.vad_detector = None
        
    def list_devices(self) -> None:
        """List available audio input devices"""
        print("\n📋 Available Audio Devices:")
        print("-" * 40)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  [{i}] {device['name']} (inputs: {device['max_input_channels']})")
        print("-" * 40)
        
    def record_fixed_duration(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Record audio for a fixed duration
        """
        duration = duration or self.default_duration
        print(f"🎤 Recording for {duration} seconds...")
        
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        )
        sd.wait()
        
        print("✅ Recording complete")
        return recording.flatten()
    
    def record_with_vad(
        self,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
        on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None,  # NEW: streaming callback
        min_speech_duration: float = 0.3,
        max_speech_duration: float = 20.0,
        silence_duration: float = 0.8,
        speech_threshold: float = 0.01,
        debug: bool = False
    ) -> np.ndarray:
        """
        Record audio using energy-based VAD
        
        Args:
            on_speech_start: Callback when speech starts
            on_speech_end: Callback when speech ends
            on_audio_chunk: Callback for each audio chunk during speech (for streaming STT)
            min_speech_duration: Minimum speech length in seconds
            max_speech_duration: Maximum recording time
            silence_duration: Silence duration to detect end of speech
            speech_threshold: Energy threshold for VAD
            debug: Print debug information
        """
        
        if self.vad_detector is None:
            self.vad_detector = VADDetector(
                sample_rate=self.sample_rate,
                threshold=speech_threshold,
                min_speech_duration_ms=int(min_speech_duration * 1000),
                silence_duration_ms=int(silence_duration * 1000)
            )
        
        # Chunk size for processing (32ms chunks)
        chunk_ms = 32
        chunk_samples = int(self.sample_rate * chunk_ms / 1000)
        
        audio_buffer = []
        
        self.is_recording = True
        self.vad_detector.reset()
        
        print("🎤 Listening... (speak when ready)")
        
        # Create a queue for thread-safe audio passing
        q = queue.Queue()
        
        def callback(indata, frames, time_info, status):
            if status and debug:
                print(f"⚠️  Audio callback status: {status}")
            q.put(indata.copy())
        
        speech_started = False
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=callback,
            dtype=np.float32,
            blocksize=chunk_samples
        ):
            start_time = time.time()
            
            while self.is_recording:
                # Check max duration
                if time.time() - start_time > max_speech_duration:
                    print("⏰ Max duration reached")
                    break
                    
                if q.empty():
                    time.sleep(0.001)  # Very short sleep
                    continue
                    
                chunk = q.get().flatten()
                
                # Ensure correct chunk size
                if len(chunk) != chunk_samples:
                    if len(chunk) < chunk_samples:
                        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                    else:
                        chunk = chunk[:chunk_samples]
                
                # Process VAD
                vad_result = self.vad_detector.process_chunk(chunk)
                state = vad_result['state']
                
                if debug:
                    energy = vad_result['energy']
                    print(f"Energy: {energy:.4f} | State: {state}")
                
                if state == 'start':
                    speech_started = True
                    if on_speech_start:
                        on_speech_start()
                    print("🗣️  Speech detected...")
                
                # Send audio chunks during speech for streaming transcription
                if self.vad_detector.triggered and on_audio_chunk:
                    on_audio_chunk(chunk)
                
                if state == 'end':
                    print("🛑 Speech ended")
                    if on_speech_end:
                        on_speech_end()
                    
                    # Return the complete utterance
                    speech_audio = vad_result['speech_audio']
                    if speech_audio is not None and len(speech_audio) > 0:
                        return speech_audio
                    else:
                        return np.array([], dtype=np.float32)
            
            # If we exit loop without 'end' state, return what we have
            if self.vad_detector.buffer:
                return np.concatenate(self.vad_detector.buffer)
        
        self.is_recording = False
        return np.array([], dtype=np.float32)

    def stop_recording(self) -> None:
        """Stop any ongoing recording"""
        self.is_recording = False
        
    def get_audio_level(self, audio: np.ndarray) -> float:
        """Get RMS energy level of audio"""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def is_speech_present(self, audio: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
        """Check if audio contains speech above threshold"""
        return self.get_audio_level(audio) > threshold
    
    def calibrate_threshold(self, duration: float = 3.0) -> float:
        """
        Calibrate the VAD threshold by measuring ambient noise
        
        Args:
            duration: Duration to record ambient noise
            
        Returns:
            Recommended threshold (2x ambient noise level)
        """
        print(f"🔇 Calibrating... Please remain silent for {duration} seconds")
        
        ambient = self.record_fixed_duration(duration)
        noise_level = self.get_audio_level(ambient)
        recommended_threshold = noise_level * 2.5
        
        print(f"   Ambient noise level: {noise_level:.4f}")
        print(f"   Recommended threshold: {recommended_threshold:.4f}")
        
        return recommended_threshold


def test_microphone():
    """Test microphone by recording and playing back"""
    print("\n🎙️  Microphone Test")
    print("=" * 40)
    
    recorder = AudioRecorder()
    recorder.list_devices()
    
    print("\nRecording 3 seconds of audio...")
    audio = recorder.record_fixed_duration(duration=3)
    
    level = recorder.get_audio_level(audio)
    print(f"Audio level: {level:.4f}")
    print(f"Speech detected: {recorder.is_speech_present(audio)}")
    
    print("\nPlaying back recording...")
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()
    
    print("✅ Microphone test complete!")


def test_vad():
    """Test Voice Activity Detection"""
    print("\n🎙️  VAD Test")
    print("=" * 40)
    
    recorder = AudioRecorder()
    
    # Calibrate threshold
    threshold = recorder.calibrate_threshold(duration=2.0)
    
    print("\nTesting VAD... Speak something!")
    audio = recorder.record_with_vad(
        speech_threshold=threshold,
        debug=True  # Show energy levels
    )
    
    if len(audio) > 0:
        print(f"\n✅ Captured {len(audio)/SAMPLE_RATE:.2f}s of speech")
        print("Playing back...")
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()
    else:
        print("\n❌ No speech detected")
    
    print("\n✅ VAD test complete!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--vad':
        test_vad()
    else:
        test_microphone()