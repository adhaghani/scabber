#!/usr/bin/env python3
"""
Voice AI Assistant - Main Entry Point

A fully local voice AI assistant for real-time voice conversations.
Combines Speech-to-Text (Whisper), LLM (Phi-3), and Text-to-Speech.

Usage:
    python main.py           # Run with default settings
    python main.py --help    # Show help
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def check_dependencies():
    """Check if required packages are installed"""
    required = [
        ('torch', 'PyTorch'),
        
        ('sounddevice', 'SoundDevice'),
        ('numpy', 'NumPy'),
        ('pyttsx3', 'pyttsx3')
    ]
    
    missing = []
    for module, name in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True


def check_audio():
    """Check if audio devices are available"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        # Check for input device
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            print("❌ No audio input devices found!")
            print("   Please connect a microphone.")
            return False
            
        print(f"✅ Found {len(input_devices)} audio input device(s)")
        return True
        
    except Exception as e:
        print(f"❌ Audio check failed: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Voice AI Assistant - Local Real-Time Conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run the voice assistant
  python main.py --check            Check system requirements
  python main.py --list-devices     List audio devices
  python main.py --test-mic         Test microphone
  python main.py --test-tts         Test text-to-speech
        """
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check system requirements'
    )
    
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio devices'
    )
    
    parser.add_argument(
        '--test-mic',
        action='store_true',
        help='Test microphone'
    )
    
    parser.add_argument(
        '--test-tts',
        action='store_true',
        help='Test text-to-speech'
    )
    
    parser.add_argument(
        '--test-stt',
        action='store_true',
        help='Test speech-to-text'
    )
    
    parser.add_argument(
        '--test-llm',
        action='store_true',
        help='Test language model'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎙️  Voice AI Assistant")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Handle different modes
    if args.check:
        print("\n📋 System Requirements Check")
        print("-" * 40)
        
        # Check audio
        check_audio()
        
        # Check GPU
        import torch
        if torch.backends.mps.is_available():
            print("✅ Apple Silicon GPU (MPS) available")
        elif torch.cuda.is_available():
            print(f"✅ CUDA GPU available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  No GPU available, will use CPU")
        
        # Check memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"✅ System memory: {mem.total / (1024**3):.1f} GB")
            print(f"   Available: {mem.available / (1024**3):.1f} GB")
        except:
            pass
            
        print("\n✅ System check complete!")
        return
        
    if args.list_devices:
        import sounddevice as sd
        print("\n📋 Audio Devices:")
        print(sd.query_devices())
        return
        
    if args.test_mic:
        from src.audio_utils import test_microphone
        test_microphone()
        return
        
    if args.test_tts:
        from src.tts_module import test_tts
        test_tts()
        return
        
    if args.test_stt:
        from src.stt_module import test_stt
        test_stt()
        return
        
    if args.test_llm:
        from src.llm_module import test_llm
        test_llm()
        return
    
    # Check audio before starting
    if not check_audio():
        sys.exit(1)
    
    # Run the assistant
    from src.voice_assistant import VoiceAssistant
    
    try:
        assistant = VoiceAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
