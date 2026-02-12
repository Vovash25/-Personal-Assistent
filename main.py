import pvrecorder
import numpy as np
import time
from speech_to_text import SpeechToText
from chatboot import Chatboot
from text_to_speech import TextToSpeech

def main():
    print("Initializing systems... Please wait.")
    stt = SpeechToText()
    bot = Chatboot()
    tts = TextToSpeech()
    
    recorder = pvrecorder.PvRecorder(device_index=0, frame_length=512)
    
    silence_threshold = 500
    silence_frames_limit = 45
    
    print("\n--- Voice Assistant Ready  ---")
    print("Commands: Say 'STOP', 'EXIT' or 'ZAKONCZ' to end the session. ")

    try:
        #nielimitowana liczba iteracji 
        while True:
            input("\n>>> Press ENTER to start talking...")
            
            recording = []
            recorder.start()
            print("Listening... (Stop speaking to finish recording)")

            silent_frames = 0
            while True:
                frame = recorder.read()
                recording.extend(frame)
                
                amplitude = np.max(np.abs(frame))
                
                if amplitude < silence_threshold:
                    silent_frames += 1
                else:
                    silent_frames = 0
                
                if silent_frames > silence_frames_limit and len(recording) > 16000:
                    break
            
            recorder.stop()
            print("Processing audio...")
            
            audio_data = np.array(recording, dtype=np.float32) / 32768.0
            
            user_text = stt(audio_data)
            if not user_text.strip():
                print("Bot: I didn't hear anything. Let's try again.")
                continue
                
            print(f"You: {user_text}")

            if any(cmd in user_text.lower() for cmd in ["stop", "exit", "zakoncz", "koniec"]):
                print("Exiting system...")
                tts("Goodbye! Do widzenia!")
                break

            response = bot(user_text)
            print(f"Bot: {response}")

            tts(response)

    except KeyboardInterrupt:
        print("\nManual shutdown.")
    finally:
        recorder.delete()

if __name__ == "__main__":
    main()