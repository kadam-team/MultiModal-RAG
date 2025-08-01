# Functions for saving and loading data to/from JSON files.

import json

def save_pages_data(data, filename):
    """Saves a list of dictionaries to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Pages data saved to '{filename}'.")
    except Exception as e:
        print(f"Error saving pages data to '{filename}': {e}")

def load_pages_data(filename):
    """Loads a list of dictionaries from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Pages data loaded from '{filename}'.")
        return data
    except FileNotFoundError:
        print(f"File '{filename}' not found. Will proceed with PDF parsing.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{filename}': {e}. File might be corrupted.")
        return None
    except Exception as e:
        print(f"Error loading pages data from '{filename}': {e}.")
        return None