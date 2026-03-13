import pandas as pd
import os

def compute_synthesized_samples_embeddings_metadata(condition_type: str, embeddings_type: str):
    
    condition_types = ["text", "audio", "text_audio"]
    for condition_type in condition_types:

        samples_metadata = f"data/TokenSynth/Samples/{condition_type}_conditioned_synthesis/{condition_type}_metadata.csv"

        # Read the samples metadata CSV file
        samples_df = pd.read_csv(samples_metadata)

        # Create a new DataFrame for embeddings metadata
        embeddings_metadata = samples_df.copy()

        # Add a column for embedding file paths
        embeddings_dir = f"data/TokenSynth/Embeddings/{condition_type}_conditioned_synthesis/{embeddings_type}_embeddings"
        embeddings_metadata["Path"] = embeddings_metadata["Path"].apply(lambda x: f"{embeddings_dir}/{os.path.basename(x).replace('/', '_').replace('.wav', '')}_embedding.pt")

        # Get the ground truth CSV
        ground_truth_path = f"data/Reymore/timber_traits_ground_truth.csv"

        # Read the ground truth CSV file
        ground_truth_df = pd.read_csv(ground_truth_path)

        # ground_truth_df columns: Instrument, RWC Name, airy-breathy, brassy-metallic...
        # For each row of embeddings metadata, find the corresponding row in ground truth depending on the Instrument and add the quality values
        # Create a mapping from RWC Name to quality values
        ground_truth_mapping = {}
        for _, row in ground_truth_df.iterrows():
            ground_truth_mapping[row["RWC Name"]] = row[2:]  # Skip first two columns (Instrument, RWC Name)

        # Add quality columns to embeddings metadata
        for quality in ground_truth_df.columns[2:]:
            embeddings_metadata[quality] = None

        # Populate quality values based on Instrument
        for idx, row in embeddings_metadata.iterrows():
            instrument = row["Instrument"]
            if instrument in ground_truth_mapping:
                for quality, value in zip(ground_truth_df.columns[2:], ground_truth_mapping[instrument]):
                    embeddings_metadata.at[idx, quality] = value

        # Save the result to a new CSV file
        embeddings_metadata_path = f"data/TokenSynth/Embeddings/{condition_type}_conditioned_synthesis/{embeddings_type}_embeddings/embeddings_labels.csv"
        embeddings_metadata.to_csv(embeddings_metadata_path, index=False)

        print(f"Embeddings metadata file saved as '{embeddings_metadata_path}'")