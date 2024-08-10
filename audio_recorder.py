import os
import pyaudio
import wave
import time
import threading


class AudioRecorder:
    def __init__(
        self,
        directory,
        device_index,
        chunk=1024,
        channels=1,
        rate=44100,
        record_seconds=10,
    ):
        self.directory = directory
        self.device_index = device_index
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self.record_seconds = record_seconds
        self.frames = []
        self.is_recording = False
        self.current_file = None

    def start(self):
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()

    def stop(self):
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()

    def _record(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk,
        )

        print("* Recording")

        start_time = int(time.time())
        while self.is_recording:
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                if not self.is_recording:
                    break
                data = stream.read(self.chunk)
                self.frames.append(data)

            end_time = int(time.time())
            self._save_file(start_time, end_time)
            self.frames = []
            start_time = end_time

        print("* Done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

    def _save_file(self, start_time, end_time):
        filename = os.path.join(self.directory, f"{start_time}-{end_time}.raw")
        with open(filename, "wb") as f:
            for frame in self.frames:
                f.write(frame)
        print(f"Saved file: {filename}")

    def combine_audio_files(self, output_file):
        raw_files = sorted(
            [f for f in os.listdir(self.directory) if f.endswith(".raw")]
        )

        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)

            for raw_file in raw_files:
                with open(os.path.join(self.directory, raw_file), "rb") as rf:
                    wf.writeframes(rf.read())

        print(f"Combined audio files into: {output_file}")
