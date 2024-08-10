import spacy
import json
from collections import defaultdict


def download_model(model_name):
    import subprocess
    import sys

    print(f"Downloading spaCy model '{model_name}'...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
    print(f"Model '{model_name}' has been successfully downloaded.")


def perform_ner(transcript_file, output_file):
    try:
        model_name = "en_core_web_lg"

        # Try to load the model, download if not available
        try:
            nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Attempting to download...")
            download_model(model_name)
            nlp = spacy.load(model_name)

        # Read the transcript
        with open(transcript_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Process the text
        doc = nlp(text)

        # Initialize a defaultdict to store entities
        entities = defaultdict(list)

        # Extract entities
        for ent in doc.ents:
            if ent.label_ in [
                "PERSON",
                "GPE",
                "LOC",
                "ORG",
                "DATE",
                "TIME",
                "MONEY",
                "PERCENT",
            ]:
                entities[ent.label_].append(ent.text)

        # Prepare the output dictionary
        output = {
            "NAMES": entities["PERSON"],
            "ADDRESSES": entities["GPE"] + entities["LOC"],
            "ORGANIZATIONS": entities["ORG"],
            "DATES": entities["DATE"],
            "TIMES": entities["TIME"],
            "MONETARY_VALUES": entities["MONEY"],
            "PERCENTAGES": entities["PERCENT"],
        }

        # Write the entities to a JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Entity recognition completed. Results saved to: {output_file}")

    except Exception as e:
        print(f"Error during entity recognition: {str(e)}")


if __name__ == "__main__":
    # This allows the module to be run standalone for testing
    import sys

    if len(sys.argv) != 3:
        print("Usage: python entity_recognition.py <transcript_file> <output_file>")
    else:
        perform_ner(sys.argv[1], sys.argv[2])
