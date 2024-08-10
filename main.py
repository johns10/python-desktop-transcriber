import os
import uuid
import pyaudio
import wave
import time
import signal
import sys
from datetime import datetime
from audio_recorder import AudioRecorder
from transcriber import transcribe_audio
from entity_recognition import perform_ner


def create_directory():
    directory = str(uuid.uuid4())
    os.makedirs(directory, exist_ok=True)
    return directory


def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get("deviceCount")

    devices = []
    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get("maxInputChannels") > 0:
            devices.append((i, device_info.get("name")))

    p.terminate()
    return devices


def select_audio_device(devices):
    print("Available audio input devices:")
    for i, (index, name) in enumerate(devices):
        print(f"{i + 1}. {name}")

    while True:
        try:
            choice = int(input("Select a device (enter the number): ")) - 1
            if 0 <= choice < len(devices):
                return devices[choice][0]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def signal_handler(sig, frame):
    print("\nExiting gracefully...")
    if "recorder" in globals():
        recorder.stop()
    sys.exit(0)


def main():
    global recorder

    signal.signal(signal.SIGINT, signal_handler)

    directory = create_directory()
    print(f"Created directory: {directory}")

    devices = list_audio_devices()
    device_index = select_audio_device(devices)

    recorder = AudioRecorder(directory, device_index)
    recorder.start()

    print("Recording... Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        combined_file = os.path.join(directory, "combined_audio.wav")
        recorder.combine_audio_files(combined_file)
        print(f"Combined audio saved to: {combined_file}")

        transcription_file = os.path.join(directory, "transcription.txt")
        transcribe_audio(combined_file, transcription_file)
        print(f"Transcription saved to: {transcription_file}")

        # Perform entity recognition
        entities_file = os.path.join(directory, "entities.json")
        perform_ner(transcription_file, entities_file)
        print(f"Entity recognition results saved to: {entities_file}")


if __name__ == "__main__":
    main()
