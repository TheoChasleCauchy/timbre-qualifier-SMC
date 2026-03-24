import os  # Operating system interfaces for directory and file operations
import pandas as pd  # Data manipulation and analysis
from tqdm import tqdm  # Progress bar for iterative tasks
import torch  # PyTorch for tensor operations and device management
import audio_to_embedding_tensor as atct  # Custom module for audio-to-embedding conversion

def compute_embeddings(embeddings_type: str):
    """
    Compute embeddings for synthesized audio samples using the specified embedding type.

    Args:
        embeddings_type (str): Type of embedding to compute (e.g., "clap", "vggish", "mert").

    This function:
    1. Iterates over each condition type (text_conditioned, audio_conditioned, text_audio_conditioned).
    2. Loads the metadata for synthesized audio samples.
    3. Constructs paths for audio samples and corresponding embedding save paths.
    4. Computes embeddings for each audio sample using the specified embedding type.
    5. Saves the computed embeddings to disk, skipping already computed embeddings.

    Steps:
    - For each condition type, load the metadata CSV file.
    - Construct full paths for audio samples and embedding save paths.
    - Initialize the audio-to-embedding converter with the specified embedding type.
    - Load all audio samples, cropping and padding to 5 seconds.
    - For each audio sample, compute its embedding and save it to disk.

    Returns:
        None: Embeddings are saved to disk in the specified directory structure.
    """
    # List of condition types to process
    condition_types = ["text_conditioned", "audio_conditioned", "text_audio_conditioned"]

    # Iterate over each condition type
    for condition_type in condition_types:
        # Path to the metadata CSV file for the current condition type
        synth_metadata_path = f"data/TokenSynth/Samples/{condition_type}_synthesis/{condition_type}_synthesis_metadata.csv"
        synth_metadata = pd.read_csv(synth_metadata_path)

        # Initialize lists to store paths for audio samples and embeddings
        samples_paths = []
        save_paths = []

        # Construct paths for audio samples and corresponding embedding save paths
        for path in synth_metadata["Path"]:
            samples_paths.append(path)
            file_name = os.path.basename(path)
            file_name = file_name.replace('.wav', '')
            save_paths.append(f"{file_name}_embedding.pt")

        # Create the output directory for embeddings if it doesn't exist
        save_dir = f"data/TokenSynth/Embeddings/{condition_type}_synthesis/{embeddings_type}_embeddings/"
        os.makedirs(save_dir, exist_ok=True)

        # Initialize the audio-to-embedding converter with the specified embedding type
        atc = atct.Audio_to_Embedding_Tensor(embedding_type=embeddings_type)

        # Load all audio samples, cropping and padding to 5 seconds
        audios = atc.load_all_audios(samples_paths, crop_to_duration=5.0, pad_to_duration=5.0)

        # Iterate over each audio sample and compute its embedding
        for indice, audio in tqdm(enumerate(audios), total=len(audios), desc=f"Computing {atc.embedding_type} embeddings"):
            # Skip if the embedding file already exists
            if os.path.exists(os.path.join(save_dir, save_paths[indice])):
                continue

            # Compute the embedding for the current audio sample
            embedding = atc.get_embedding(audio)

            # Save the embedding to disk
            torch.save(embedding.cpu(), os.path.join(save_dir, save_paths[indice]))
