# Scabber — Instant TTS (pocket-tts)

Short description
- Scabber provides an instant, local text-to-speech (TTS) machine powered by `pocket-tts`.
- The repository includes a `backend/` service that exposes a simple interface to synthesize speech from text.

Quickstart
1. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r backend/requirements.txt
   ```

3. Start the backend service (example):

   ```bash
   python backend/server.py
   # or, if the project uses FastAPI / uvicorn:
   # uvicorn backend.main:app --reload
   ```

Usage
- Send text to the TTS service and save the audio output. A common pattern is an HTTP POST request with JSON containing the text to synthesize. Example (adjust the URL/path as needed for this repo):

  ```bash
  curl -X POST http://localhost:8000/synthesize \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello world"}' \
    --output hello.wav
  ```

- The service will return raw audio (e.g., WAV) or a URL to the generated file depending on configuration.

Project layout (important parts)
- `backend/`: server code, TTS integration and API endpoints.
- `data/`, `models/`: local data and model artifacts used by the TTS engine.

Notes & troubleshooting
- Ensure `pocket-tts` and any required model files are available in the environment.
- If audio is silent or playback fails, check sample rate/codec compatibility in the client player.
- If the backend exposes a different endpoint or auth, consult the backend module that implements the TTS route.

Development
- Add voices, tweak synthesis parameters, or wire the TTS output into other services (web UI, CLI, or messaging bots).

Contributing
- Open an issue or PR with reproducible steps and expected behavior.

License
- See repository root for license details (if provided).

Contact
- For questions about this README or running the service, open an issue in this repository.
