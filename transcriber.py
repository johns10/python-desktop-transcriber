import whisperx
import torch


def transcribe_audio(audio_file, output_file):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 8
        compute_type = "int8"
        print(f"Using device: {device}")
        model = whisperx.load_model("base", device, compute_type=compute_type)
        result = model.transcribe(audio_file, batch_size=batch_size)

        with open(output_file, "w") as f:
            for segment in result["segments"]:
                f.write(
                    f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}\n"
                )

        print(f"Transcription completed and saved to: {output_file}")
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
