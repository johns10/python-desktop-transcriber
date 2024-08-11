import whisperx
import torch
import os


def transcribe_audio(audio_file, output_file):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 8
        compute_type = "int8"
        print(f"Using device: {device}")

        # Load ASR model
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)

        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=os.environ["HUGGINGFACE_API_KEY"], device=device
        )
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        json_data = []
        with open(output_file, "w") as f:
            for segment in result["segments"]:
                start_time = f"{segment['start']:.2f}"
                end_time = f"{segment['end']:.2f}"

                # Check if speaker has changed
                if segment["speaker"] != current_speaker:
                    current_speaker = segment["speaker"]
                    f.write(f"\n[Speaker {current_speaker}]\n")

                f.write(f"[{start_time}s - {end_time}s] {segment['text']}\n")

                # Add segment info to JSON data
                json_data.append(
                    {
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "speaker": current_speaker,
                        "text": segment["text"],
                    }
                )

        print(
            f"Transcription with speaker diarization completed and saved to: {output_file}"
        )

        # Write JSON file
        json_file = os.path.splitext(output_file)[0] + ".json"
        with open(json_file, "w") as jf:
            json.dump(json_data, jf, indent=2)

        print(f"JSON data with speaker information saved to: {json_file}")

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
