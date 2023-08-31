import pyaudio
import wave


# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sample rate (samples per second)
CHUNK = 1024  # Number of frames per buffer
FRAMES_PER_BUFFER = 3200

# Create an instance of the PyAudio class
p = pyaudio.PyAudio()

# Open a stream for audio input
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
    )

print("Recording... (press Ctrl+C to stop)")

frames = []

try:
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
except KeyboardInterrupt:
    print("Saving to file...")
    pass

# Close and terminate the audio stream and PyAudio instance
stream.stop_stream()
stream.close()
p.terminate()

# Save recorded frames to a WAV file
output_file = "recorded_audio.wav"
with wave.open(output_file, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print("Recording saved as", output_file)
