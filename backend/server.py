import asyncio
import json
import logging
import sys
import os
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.voice_assistant import VoiceAssistant
from config import DEVICE, SAMPLE_RATE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceAssistantServer")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("Client connected")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("Client disconnected")

    async def send_text(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_bytes(self, data: bytes, websocket: WebSocket):
        await websocket.send_bytes(data)

manager = ConnectionManager()

# Initialize Voice Assistant (Singleton-ish for simplicity)
# We need to adapt VoiceAssistant to be more server-friendly (non-blocking)
# reusing the existing class structure but bypassing the run loop
assistant = VoiceAssistant()
assistant.load_models()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Initialize VAD for this connection
    from src.audio_utils import VADDetector
    vad = VADDetector(
        sample_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=300,
        silence_duration_ms=800
    )
    
    # Audio buffer to hold the *original* high-quality audio
    # Note: If client sends float32 at 16k, we are good.
    # If client sends something else, we might need resampling.
    # We'll assume client sends 16k Float32 for now as per `audio-stream.ts` default.
    audio_buffer = []
    
    try:
        while True:
            data = await websocket.receive()
            
            if "bytes" in data:
                audio_bytes = data["bytes"]
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Check for VAD
                # VADDetector expects ~32ms chunks at 16k (512 samples)
                # The browser/frontend script `audio-stream.ts` buffer size is 4096.
                # 4096 / 16000 = ~256ms. This is too large for single VAD step?
                # Actually Silero can handle larger chunks, but standard is small.
                # Let's slice it or pass it through. `process_chunk` logic in VADDetector
                # handles the VAD call. We updated `process_chunk` to take any size 
                # but let's check VADDetector implementation again. 
                # My implementation of process_chunk takes `chunk`, calculates `chunk_ms`,
                # and runs model. Silero model forward pass takes the whole tensor.
                # So passing 256ms chunk is fine, it will return prob for that chunk.
                
                vad_result = vad.process_chunk(audio_array)
                state = vad_result['state']
                
                if state == 'start':
                    # Speech started in this chunk
                    logger.info("🗣️ Speech detected...")
                    # Append this chunk to buffer
                    audio_buffer.append(audio_array)
                    
                elif state == 'speech':
                    # Continuing speech
                    audio_buffer.append(audio_array)
                    
                elif state == 'end':
                    # Speech ended
                    logger.info("mute Speech ended")
                    # Append final chunk? VADDetector buffers internally but we want ours.
                    audio_buffer.append(audio_array)
                    
                    # Process the complete utterance
                    if audio_buffer:
                        full_audio = np.concatenate(audio_buffer)
                        audio_buffer = [] # Clear buffer
                        vad.reset() # Reset VAD state for next utterance
                        
                        # Process asynchronous to not block WS loop?
                        # Ideally yes, but for now blocking is okay for single user.
                        
                        # Processing Logic
                        # STT
                        user_text = assistant.stt.transcribe(full_audio)
                        if user_text and len(user_text.strip()) >= 2:
                            logger.info(f"User: {user_text}")
                            await manager.send_text(json.dumps({"type": "transcription", "text": user_text}), websocket)
                            
                            # LLM
                            response_text = assistant.llm.generate_response(user_text)
                            logger.info(f"Assistant: {response_text}")
                            await manager.send_text(json.dumps({"type": "response", "text": response_text}), websocket)
                            
                            # TTS
                            audio_response = assistant.tts.generate_audio_bytes(response_text)
                            if audio_response is not None:
                                await manager.send_bytes(audio_response.tobytes(), websocket)
                
                elif state == 'silence':
                    # If we have triggered, we are waiting for end or more speech
                    if vad.triggered:
                         audio_buffer.append(audio_array)
                    else:
                        # Just silence, ignore
                        pass

            elif "text" in data:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
