import whisperx
import torch


def transcribe_audio(audio_file, output_file):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 8
        compute_type = "int8"
        print(f"Using device: {device}")

        # Load ASR model
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)

        # Transcribe audio
        result = model.transcribe(audio_file, batch_size=batch_size)

        # Load diarization model
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device=device)

        # Perform diarization
        diarize_segments = diarize_model(audio_file)

        # Assign speaker labels
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Write results to file
        with open(output_file, "w") as f:
            current_speaker = None
            for segment in result["segments"]:
                start_time = f"{segment['start']:.2f}"
                end_time = f"{segment['end']:.2f}"

                # Check if speaker has changed
                if segment["speaker"] != current_speaker:
                    current_speaker = segment["speaker"]
                    f.write(f"\n[Speaker {current_speaker}]\n")

                f.write(f"[{start_time}s - {end_time}s] {segment['text']}\n")

        print(
            f"Transcription with speaker diarization completed and saved to: {output_file}"
        )
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
