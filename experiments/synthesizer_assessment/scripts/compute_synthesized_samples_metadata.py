import os  # Operating system interfaces for directory and file operations
import pandas as pd  # Data manipulation and analysis

def compute_synth_metadata():
    """
    Generate metadata for synthesized audio samples.

    This function:
    1. Iterates over each condition type (text, audio, text_audio).
    2. Walks through the directory structure of synthesized audio samples.
    3. Collects file paths and corresponding instrument names.
    4. Creates a DataFrame with the collected metadata.
    5. Saves the metadata to a CSV file for each condition type.

    Steps:
    - For each condition type, traverse the directory containing synthesized audio samples.
    - Collect the paths of all WAV files and their corresponding instrument names.
    - Create a DataFrame with the collected metadata.
    - Save the DataFrame to a CSV file in the same directory as the audio samples.

    Returns:
        None: Metadata files are saved to disk.
    """
    # List of condition types to process
    condition_types = ["text", "audio", "text_audio"]

    # Iterate over each condition type
    for condition_type in condition_types:
        # Path to the directory containing synthesized audio samples for the current condition type
        audios_folder = f"data/TokenSynth/Samples/{condition_type}_conditioned_synthesis/"

        # Initialize lists to store file paths and instrument names
        paths = []
        instruments = []

        # Walk through the directory structure to find all WAV files
        for root, _, files in os.walk(audios_folder):
            for file in files:
                # Check if the file is a WAV file
                if file.endswith('.wav'):
                    # Get the full path of the WAV file
                    full_path = os.path.join(root, file)
                    paths.append(full_path)

                    # Get the instrument name from the subfolder name
                    subfolder = os.path.basename(root)
                    instruments.append(subfolder)

        # Create a DataFrame with the collected metadata
        df = pd.DataFrame({
            'Path': paths,
            'Instrument': instruments
        })

        # Create the output directory if it doesn't exist
        output_dir = f"data/TokenSynth/Samples/{condition_type}_conditioned_synthesis/"
        os.makedirs(output_dir, exist_ok=True)

        # Save the metadata DataFrame to a CSV file
        df.to_csv(f"{output_dir}{condition_type}_conditioned_synthesis_metadata.csv", index=False)

        # Print a message indicating the location of the saved metadata file
        print(f"Samples metadata file saved as '{output_dir}{condition_type}_conditioned_synthesis_metadata.csv'")
