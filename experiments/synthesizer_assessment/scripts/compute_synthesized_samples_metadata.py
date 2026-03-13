import os
import pandas as pd

def compute_synth_metadata():

    condition_types = ["text", "audio", "text_audio"]
    for condition_type in condition_types:

        audios_folder = f"data/TokenSynth/Samples/{condition_type}_conditioned_synthesis/"

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
        output_dir = f"data/TokenSynth/Samples/{condition_type}_conditioned_synthesis/"
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(f"{output_dir}{condition_type}_metadata.csv", index=False)

        print(f"Samples metadata file saved as '{output_dir}{condition_type}_metadata.csv'")