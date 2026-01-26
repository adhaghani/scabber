"""
Main Voice Assistant class that integrates all modules
With real-time VAD and streaming responses
"""
import sys
import os
from typing import Optional
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DEVICE

from .stt_module import SpeechToText
from .llm_module import LanguageModel  
from .tts_module import TextToSpeech
from .audio_utils import AudioRecorder


class VoiceAssistant:
    """
    Voice AI Assistant - integrates STT, LLM, and TTS for 
    real-time voice conversations with streaming responses
    """
    
    def __init__(self):
        print("🚀 Initializing Voice Assistant...")
        print(f"   Device: {DEVICE}")
        print("=" * 50)
        
        # Initialize components
        self.audio_recorder = AudioRecorder()
        self.stt = SpeechToText()
        self.llm = LanguageModel()
        self.tts = TextToSpeech(voice="alba")
        
        # State
        self.is_running = False
        self.use_realtime_mode = True
        self.use_streaming = False
        self.auto_smart_mode = True
        self.pending_smart_question = None
        
        # Calibrated VAD threshold (will be set during initialization)
        self.vad_threshold = None
        
    def load_models(self) -> None:
        """Load all AI models"""
        print("\n📦 Loading Models...")
        print("-" * 50)
        
        # Load STT (Kyutai)
        self.stt.load()
        
        # Load LLM
        self.llm.load()
        
        # Load TTS
        self.tts.load()
        
        print("-" * 50)
        print("✅ All models loaded!")
        
    def calibrate_microphone(self) -> None:
        """Calibrate the microphone threshold"""
        print("\n🎤 Microphone Calibration")
        print("-" * 50)
        print("Please remain silent for 2 seconds while I calibrate...")
        
        self.vad_threshold = self.audio_recorder.calibrate_threshold(duration=2.0)
        
        print(f"✅ Calibration complete! Threshold set to: {self.vad_threshold:.4f}")
        print("-" * 50)
        
    def process_audio(self, audio) -> Optional[str]:
        """
        Process audio through the full pipeline
        
        Args:
            audio: numpy array of audio samples
            
        Returns:
            Assistant's spoken response text, or None if no speech detected
        """
        if len(audio) == 0:
            return None
            
        # Transcribe
        print("🔄 Transcribing...")
        user_text = self.stt.transcribe(audio)
        
        if not user_text or len(user_text.strip()) < 2:
            return None
            
        print(f"👤 You: {user_text}")
        
        # Check for exit commands
        exit_words = ['quit', 'exit', 'goodbye', 'bye', 'stop']
        if any(word in user_text.lower() for word in exit_words):
            response = "Goodbye! It was nice talking to you."
            self.speak(response)
            self.is_running = False
            return response
        
        # Handle smart model confirmation responses
        if self.pending_smart_question:
            return self._handle_smart_model_confirmation(user_text)
        
        # Check if user wants to exit smart mode
        if self.llm.is_using_smart_model():
            return self._handle_smart_mode_question(user_text)
        
        # Check if question is complex and offer smart model
        if self.auto_smart_mode and self.llm.is_complex_question(user_text):
            return self._offer_smart_model(user_text)
        
        # Generate and speak response with normal model
        if self.use_streaming:
            response = self.process_streaming(user_text)
        else:
            print("🤔 Thinking...")
            response = self.llm.generate_response(user_text)
            self.speak(response)
        
        return response
    
    def _offer_smart_model(self, user_text: str) -> str:
        """
        Offer to use the smart model for a complex question
        
        Args:
            user_text: The user's complex question
            
        Returns:
            Response asking if user wants smart model
        """
        self.pending_smart_question = user_text
        
        response = "That sounds like a thoughtful question. Would you like me to use my deeper thinking mode for a more detailed answer? Just say yes or no."
        self.speak(response)
        return response
    
    def _handle_smart_model_confirmation(self, user_text: str) -> str:
        """
        Handle user's response to smart model offer
        
        Args:
            user_text: User's yes/no response
            
        Returns:
            Response from appropriate model
        """
        user_lower = user_text.lower().strip()
        original_question = self.pending_smart_question
        self.pending_smart_question = None
        
        # Check for affirmative responses
        yes_words = ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'please', 'go ahead', 'do it', 'yes please']
        no_words = ['no', 'nope', 'nah', 'don\'t', 'skip', 'nevermind', 'never mind', 'no thanks']
        
        if any(word in user_lower for word in yes_words):
            # User wants smart model
            return self._process_with_smart_model(original_question)
        elif any(word in user_lower for word in no_words):
            # User prefers quick answer
            response = "No problem, let me give you a quick answer."
            self.speak(response)
            
            print("🤔 Thinking...")
            response = self.llm.generate_response(original_question)
            self.speak(response)
            return response
        else:
            # Unclear response, ask again
            self.pending_smart_question = original_question
            response = "I didn't catch that. Would you like me to use my deeper thinking mode? Say yes or no."
            self.speak(response)
            return response
    
    def _process_with_smart_model(self, user_text: str) -> str:
        """
        Process question with the smart model
        
        Args:
            user_text: The user's question
            
        Returns:
            Response from smart model
        """
        # Speak thinking phrase while loading
        thinking_phrase = self.llm.get_thinking_phrase()
        self.speak(thinking_phrase)
        
        # Load smart model
        print("🧠 Loading smart model...")
        if self.llm.load_smart_model():
            # Generate response with smart model
            print("🤔 Deep thinking...")
            response = self.llm.generate_response(user_text)
            self.speak(response)
            
            # Ask if user has more complex questions
            followup = "Would you like to ask another detailed question while I'm in deep thinking mode? Say yes to continue, or no to switch back to quick mode."
            self.speak(followup)
            
            return response
        else:
            # Failed to load smart model, use base model
            response = "I couldn't load my deeper thinking mode, but let me answer with what I have."
            self.speak(response)
            
            response = self.llm.generate_response(user_text)
            self.speak(response)
            return response
    
    def _handle_smart_mode_question(self, user_text: str) -> str:
        """
        Handle questions while in smart mode
        
        Args:
            user_text: The user's input
            
        Returns:
            Response
        """
        user_lower = user_text.lower().strip()
        
        # Check if user wants to exit smart mode
        no_words = ['no', 'nope', 'nah', 'done', 'finished', 'that\'s all', 'switch back', 'quick mode', 'no thanks']
        yes_words = ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'another', 'one more']
        
        if any(word in user_lower for word in no_words):
            # User wants to exit smart mode
            self.llm.unload_smart_model()
            response = "Okay, I've switched back to quick response mode. What else can I help you with?"
            self.speak(response)
            return response
        elif any(word in user_lower for word in yes_words) and len(user_lower.split()) < 5:
            # User confirmed they have another question
            response = "Great, go ahead and ask your question."
            self.speak(response)
            return response
        else:
            # Treat it as a new question - answer with smart model
            print("🤔 Deep thinking...")
            response = self.llm.generate_response(user_text)
            self.speak(response)
            
            # Ask if user has more complex questions
            followup = "Do you have another detailed question, or should I switch back to quick mode?"
            self.speak(followup)
            
            return response
    
    def process_streaming(self, user_text: str) -> str:
        """
        Generate streaming response and speak it in chunks
        
        Args:
            user_text: Transcribed user speech
            
        Returns:
            Full response text
        """
        print("🤔 ", end="", flush=True)
        
        full_response = ""
        sentence_buffer = ""
        
        # Stream response from LLM
        for chunk in self.llm.generate_response_stream(user_text):
            print(chunk, end="", flush=True)
            full_response += chunk
            sentence_buffer += chunk
            
            # Speak complete sentences as they arrive
            if any(punct in sentence_buffer for punct in ['.', '!', '?', '\n']):
                # Find the last sentence boundary
                for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                    if punct in sentence_buffer:
                        idx = sentence_buffer.rfind(punct)
                        sentence = sentence_buffer[:idx + 1].strip()
                        sentence_buffer = sentence_buffer[idx + len(punct):]
                        
                        if sentence:
                            # Speak this sentence (non-blocking to continue streaming)
                            self.tts.speak(sentence)
                        break
        
        print()  # Newline after streaming
        
        # Speak any remaining text
        if sentence_buffer.strip():
            self.tts.speak(sentence_buffer.strip())
        
        print(f"🤖 Assistant: {full_response}")
        return full_response
        
    def speak(self, text: str) -> None:
        """Speak text using TTS"""
        print(f"🤖 Assistant: {text}")
        self.tts.speak(text)
        
    def record_realtime(self) -> Optional[str]:
        """Record with real-time VAD and process"""
        # Use calibrated threshold or default
        threshold = self.vad_threshold if self.vad_threshold else 0.01
        
        audio = self.audio_recorder.record_with_vad(
            min_speech_duration=0.3,
            max_speech_duration=30.0,
            silence_duration=0.6,  # Shorter for faster response
            speech_threshold=threshold,  # ✅ Use calibrated or default threshold
            debug=False  # Set to True to see energy levels
        )
        return self.process_audio(audio)
    
    def record_and_process(self) -> Optional[str]:
        """Record audio and process it (legacy mode)"""
        audio = self.audio_recorder.record_fixed_duration()
        return self.process_audio(audio)
    
    def run_continuous(self) -> None:
        """Continuous listening mode - no Enter required"""
        print("\n🔄 Continuous listening mode active!")
        print("   I'm always listening - just speak naturally.")
        print("   Say 'quit', 'exit', or 'goodbye' to stop.")
        print("   Press Ctrl+C to force exit.\n")
        
        while self.is_running:
            try:
                # Continuously listen with VAD
                response = self.record_realtime()
                
                if response is None:
                    # No speech detected, just keep listening
                    pass
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
    def run(self) -> None:
        """Main conversation loop"""
        # Load models first
        self.load_models()
        
        # Calibrate microphone
        self.calibrate_microphone()
        
        print("\n" + "=" * 60)
        print("🎙️  Voice Assistant Ready!")
        print("=" * 60)
        print("\nModes:")
        print("  • continuous - Always listening (no Enter needed)")
        print("  • realtime   - VAD detection, press Enter to start")
        print("  • manual     - 5 second fixed recording")
        print("\nCommands:")
        print("  • quit      - Exit the assistant")
        print("  • clear     - Clear conversation history")
        print("  • voices    - List available TTS voices")
        print("  • stream    - Toggle streaming responses")
        print("  • smart     - Toggle auto smart mode for complex questions")
        print("  • calibrate - Recalibrate microphone threshold")
        print("  • test      - Test microphone with current threshold")
        print("=" * 60)
        
        # Mode selection
        mode = input("\nChoose mode (Enter for continuous, 'realtime', or 'manual'): ").strip().lower()
        
        if mode == 'continuous' or mode == '':
            self.use_realtime_mode = True
            self.is_running = True
            self.run_continuous()
            self.cleanup()
            return
        
        self.use_realtime_mode = mode != 'manual'
        
        if self.use_realtime_mode:
            print("\n📢 Real-time mode enabled!")
            print("   Press Enter, then speak - I'll detect when you're done.")
        else:
            print("\n📢 Manual mode enabled!")
            print("   Press Enter to start 5-second recording.")
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Wait for user to be ready
                user_input = input("\n[Enter to speak | quit | clear | voices | stream | smart | calibrate | test]: ").strip().lower()
                
                if user_input == 'quit':
                    self.speak("Goodbye! It was nice talking to you.")
                    break
                elif user_input == 'clear':
                    self.llm.clear_history()
                    if self.llm.is_using_smart_model():
                        self.llm.unload_smart_model()
                    print("🗑️  Conversation history cleared")
                    continue
                elif user_input == 'voices':
                    self.tts.list_voices()
                    continue
                elif user_input == 'stream':
                    self.use_streaming = not self.use_streaming
                    status = "enabled" if self.use_streaming else "disabled"
                    print(f"📡 Streaming responses {status}")
                    continue
                elif user_input == 'smart':
                    self.auto_smart_mode = not self.auto_smart_mode
                    status = "enabled" if self.auto_smart_mode else "disabled"
                    print(f"🧠 Auto smart mode {status}")
                    if not self.auto_smart_mode and self.llm.is_using_smart_model():
                        self.llm.unload_smart_model()
                    continue
                elif user_input == 'calibrate':
                    self.calibrate_microphone()
                    continue
                elif user_input == 'test':
                    print("\n🎤 Testing microphone...")
                    print("Speak something for 3 seconds...")
                    test_audio = self.audio_recorder.record_fixed_duration(duration=3)
                    level = self.audio_recorder.get_audio_level(test_audio)
                    threshold = self.vad_threshold if self.vad_threshold else 0.01
                    print(f"   Audio level: {level:.4f}")
                    print(f"   Current threshold: {threshold:.4f}")
                    print(f"   Speech detected: {'YES ✅' if level > threshold else 'NO ❌'}")
                    if level < threshold:
                        print("   ⚠️  Your voice is below the threshold!")
                        print("      Try speaking louder or recalibrating with 'calibrate'")
                    continue
                elif user_input == 'continuous':
                    self.run_continuous()
                    break
                elif user_input == 'realtime':
                    self.use_realtime_mode = True
                    print("📢 Switched to real-time VAD mode!")
                    continue
                elif user_input == 'manual':
                    self.use_realtime_mode = False
                    print("📢 Switched to manual 5-second mode!")
                    continue
                
                # Record and process based on mode
                if self.use_realtime_mode:
                    response = self.record_realtime()
                else:
                    response = self.record_and_process()
                
                if response is None:
                    print("❌ No speech detected. Try again.")
                    print("   Tip: Type 'test' to check your microphone level")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        self.cleanup()
        
    def cleanup(self) -> None:
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        # Unload smart model if active
        if self.llm.is_using_smart_model():
            self.llm.unload_smart_model()
        self.stt.unload()
        self.llm.unload()
        self.tts.unload()
        print("✅ Cleanup complete!")


def main():
    """Entry point"""
    try:
        assistant = VoiceAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()