import json
import os
import pandas as pd
# Adjust file path to point to `embeddings.json` within your project directory
STORAGE_FILE = os.path.join(os.path.dirname(__file__), "embeddings.json")

def store_embeddings(embeddings):
    """Stores the provided DataFrame of embeddings to a JSON file."""
    try:
        with open(STORAGE_FILE, "w") as f:
            json.dump(embeddings.to_dict(), f, indent=4)  # Formatted JSON for readability
    except Exception as e:
        print(f"Error storing embeddings: {e}")

def load_embeddings():
    """Loads embeddings from the JSON file if it exists.

    Returns the DataFrame of embeddings if found, None otherwise.
    """
    try:
        with open(STORAGE_FILE, "r") as f:
            return pd.DataFrame(json.load(f))
    except FileNotFoundError:
        # File not found, return None
        return None
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None
