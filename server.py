from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps
from faster import NavaiFastSTT

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

vad_model = load_silero_vad()
transcriber = NavaiFastSTT()

SAMPLE_RATE = 16000
MAX_SILENCE_BLOCKS = 3

# Shared state
audio_buffer = np.zeros(0, dtype=np.float32)
silence_blocks = 0

@app.route("/")
def index():
    return render_template("index.html")

def transcribe_and_emit(buffer_copy):
    text = transcriber.transcribe(buffer_copy)
    socketio.emit("text", {"text": text})

@socketio.on("audio")
def handle_audio(data):
    global audio_buffer, silence_blocks

    chunk = np.frombuffer(data, dtype=np.float32)
    speech_ts = get_speech_timestamps(
        chunk, vad_model, sampling_rate=SAMPLE_RATE, return_seconds=True
    )

    if speech_ts:
        audio_buffer = np.concatenate((audio_buffer, chunk))
        silence_blocks = 0
    else:
        if audio_buffer.size > 0:
            audio_buffer = np.concatenate((audio_buffer, chunk))
            silence_blocks += 1

            if silence_blocks >= MAX_SILENCE_BLOCKS:
                buffer_copy = audio_buffer.copy()
                audio_buffer = np.zeros(0, dtype=np.float32)
                silence_blocks = 0
                socketio.start_background_task(transcribe_and_emit, buffer_copy)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
