"""
Configuration settings for Voice AI Assistant
"""
import os
from pathlib import Path

# Load .env file if present (never overrides already-exported shell vars)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed yet; rely on shell env vars

# Model Settings
WHISPER_MODEL_ID = "tiny"  # OpenAI Whisper model (offline)
LLM_MODEL = "arcee-ai/trinity-mini:free"  # Default model displayed in logs
OPENROUTER_MODEL = "arcee-ai/trinity-mini:free"  # Fast model for simple questions
SMART_MODEL = "arcee-ai/trinity-mini:free"  # Smarter model for complex questions
USE_OPENROUTER = True  # Use OpenRouter instead of HuggingFace transformers

# OpenRouter API Settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "Scabber Voice Assistant")

# Complexity Detection - keywords that suggest a complex question
COMPLEXITY_KEYWORDS = []

# Thinking phrases for the small model while loading smart model
THINKING_PHRASES = [
    "Let me think about this carefully.",
    "This is an interesting question. Give me a moment to consider it properly.",
    "Hmm, that's a great question. Let me bring in some deeper thinking.",
    "I want to give you a thorough answer. Just a moment.",
    "This deserves careful consideration. One moment please."
]

# Model Cache Directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Audio Settings
SAMPLE_RATE = 16000  # Recording sample rate
STT_SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1
RECORDING_DURATION = 5  # seconds
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 4.0  # seconds of silence to stop recording

# Device Settings
def get_device():
    """Get the best available device"""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()

# Memory Settings
USE_INT8_QUANTIZATION = True
LOW_CPU_MEM_USAGE = True
TORCH_DTYPE = "float16"

# Conversation Settings
MAX_NEW_TOKENS = 512  # Shorter responses = faster (for base model)
SMART_MODEL_MAX_TOKENS = 1024  # Longer responses for smart model
TEMPERATURE = 0.5
TOP_P = 0.9

# TTS Settings for audio generation
TTS_MAX_TOKENS = 2048  # Max audio tokens per utterance (~160 seconds max)

# System Prompt for base model (concise responses)
SYSTEM_PROMPT = """You are an AI assistant engaged in a voice conversation. 
Keep your responses concise, as they will be spoken aloud.
Be helpful and natural in your responses. only ask further clarification or questions when asked to.
IMPORTANT: Never use emojis, asterisks, markdown formatting, or special symbols in your responses. Never use questions like "please list" or anything that is hard to do via voice.
Use plain AND Spaces text only."""

# System Prompt for smart model (detailed, thorough responses)
SMART_SYSTEM_PROMPT = """You are an expert AI assistant in deep thinking mode, engaged in a voice conversation.
The user has specifically requested a detailed, thorough answer. Take your time to fully explain the topic.

Guidelines for your response:
- Provide comprehensive, in-depth explanations
- Break down complex topics into clear sections
- Include relevant examples, analogies, or step-by-step breakdowns
- Cover multiple angles or perspectives when appropriate
- Anticipate follow-up questions and address them proactively
- Aim for educational, substantive responses that fully satisfy curiosity

IMPORTANT: Never use emojis, asterisks, markdown formatting, or special symbols. Use plain text with clear paragraph breaks only."""
