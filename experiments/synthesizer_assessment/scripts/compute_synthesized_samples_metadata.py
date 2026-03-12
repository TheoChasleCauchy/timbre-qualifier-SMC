import os
import pandas as pd

def compute_synth_metadata(audios_folder: str, csv_name: str):

    condition_types = ["text", "audio", "text_audio"]
    for condition_type in condition_types:

        # Initialize lists to store paths and instrument names
        paths = []
        instruments = []

        # Walk through the parent folder
        for root, _, files in os.walk(audios_folder):
            for file in files:
                if file.endswith('.wav'):
                    # Get the full path of the .wav file
                    full_path = os.path.join(root, file)
                    paths.append(full_path)

                    # Get the name of the subfolder (Instrument)
                    subfolder = os.path.basename(root)
                    instruments.append(subfolder)

        # Create a DataFrame
        df = pd.DataFrame({
            'Path': paths,
            'Instrument': instruments
        })

        # Write to CSV
        os.makedirs("./resources/metadata/Synth/", exist_ok=True)
        df.to_csv(f"./resources/metadata/Synth/{csv_name}", index=False)

        print(f"CSV file saved as './resources/metadata/Synth/{csv_name}'")