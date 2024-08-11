import spacy
import json
from collections import defaultdict
from flair.data import Sentence
from flair.models import SequenceTagger
import re


def download_model(model_name):
    import subprocess
    import sys

    print(f"Downloading spaCy model '{model_name}'...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
    print(f"Model '{model_name}' has been successfully downloaded.")


def extract_address(spans):
    address_components = defaultdict(list)
    for span in spans:
        if span.tag in [
            "Building_Number",
            "Street_Name",
            "City",
            "Province",
            "Postal_Code",
            "Country",
        ]:
            address_components[span.tag].append(span.text)

    address_parts = []
    for tag in [
        "Building_Number",
        "Street_Name",
        "City",
        "Province",
        "Postal_Code",
        "Country",
    ]:
        if address_components[tag]:
            address_parts.append(max(address_components[tag], key=len))

    return " ".join(address_parts).strip()


def perform_ner(transcript_json, output_file):
    try:
        model_name = "en_core_web_lg"

        # Try to load the spaCy model, download if not available
        try:
            nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Attempting to download...")
            download_model(model_name)
            nlp = spacy.load(model_name)

        # Load the custom Canadian address NER model
        canadian_ner_model = SequenceTagger.load(
            "./resources/canadian-address-extraction.pt"
        )

        # Read the transcript JSON
        with open(transcript_json, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        # Initialize a defaultdict to store entities
        entities = defaultdict(list)

        # Process each segment in the transcript
        for segment in transcript_data:
            # Reconstruct the text from word-level data
            text = " ".join([word["word"] for word in segment["words"]])

            # Process the text with spaCy
            doc = nlp(text)

            # Extract entities using spaCy
            for ent in doc.ents:
                if ent.label_ in [
                    "PERSON",
                    "ORG",
                    "DATE",
                    "TIME",
                    "MONEY",
                    "PERCENT",
                ]:
                    entities[ent.label_].append(ent.text)

            # Process the text with the Canadian address NER model
            sentence = Sentence(text)
            canadian_ner_model.predict(sentence)

            # Extract Canadian addresses
            print(sentence.get_spans("ner"))
            address = extract_address(sentence.get_spans("ner"))
            if address:
                entities["ADDRESS"].append(address)

        # Prepare the output dictionary
        output = {
            "NAMES": entities["PERSON"],
            "ADDRESSES": entities["ADDRESS"],
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
        print("Usage: python entity_recognition.py <transcript_json> <output_file>")
    else:
        perform_ner(sys.argv[1], sys.argv[2])
