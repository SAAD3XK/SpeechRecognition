import speech_recognition
import pyttsx3
import logging

# logging.basicConfig(level=logging.info)

recognizer_instance = speech_recognition.Recognizer()

while True:
    try:
        with speech_recognition.Microphone() as mic:
            recognizer_instance.adjust_for_ambient_noise(mic, duration=0.5)
            # recognizer_instance.energy_threshold=500
            # recognizer_instance.dynamic_energy_threshold=False
            audio = recognizer_instance.listen(mic)
            text = recognizer_instance.recognize_whisper(audio, language='english')
            # text = recognizer_instance.recognize_whisper(audio, language='urdu', translate=True)
            
            print(f"Recognized text: {text}")
        
    except speech_recognition.UnknownValueError():
        recognizer_instance = speech_recognition.Recognizer()
        continue