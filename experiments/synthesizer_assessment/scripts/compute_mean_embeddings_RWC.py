from tokensynth import CLAP  # CLAP model for audio encoding
import pandas as pd  # Data manipulation and analysis
from tqdm import tqdm  # Progress bar for iterative tasks
import os  # Operating system interfaces for directory and file operations
import torch  # PyTorch for tensor operations and device management

def compute_mean_embeddings():
    """
    Compute mean embeddings for each instrument in the RWC dataset using the CLAP model.

    This function:
    1. Initializes the CLAP model for audio encoding.
    2. Loads the RWC metadata CSV file.
    3. Groups the metadata by instrument.
    4. For each instrument, computes the mean embedding of all its audio samples.
    5. Saves the mean embeddings to disk as PyTorch tensor files.

    Steps:
    - Initialize the CLAP model and set the device (GPU if available, else CPU).
    - Load the RWC metadata from a CSV file.
    - Group the metadata by instrument.
    - For each instrument, iterate over its audio samples, compute embeddings, and calculate the mean.
    - Save the mean embeddings to disk.

    Returns:
        None: Mean embeddings are saved to disk in the specified directory structure.
    """
    # Initialize the CLAP model and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clap = CLAP(device=device)

    # Paths for metadata, audio files, and output embeddings
    metadata_path = "data/metadata/RWC/RWC_metadata.csv"
    audios_path = "data/RWC/RWC-preprocessed/"
    embeddings_path = f"data/RWC/mean_clap_embeddings/"
    os.makedirs(embeddings_path, exist_ok=True)  # Create the output directory if it doesn't exist

    # Load the RWC metadata
    metadata = pd.read_csv(metadata_path)

    # Group the metadata by instrument
    instrument_groups = metadata.groupby("Instrument")

    # Iterate over each instrument group
    for instrument, instrument_df in instrument_groups:
        # Skip if the mean embedding for this instrument already exists
        if os.path.exists(f"{embeddings_path}{instrument}_embedding.pt"):
            continue

        # Disable gradient computation for efficiency
        with torch.no_grad():
            embeddings = []
            # Iterate over each audio sample for the current instrument
            for _, row in tqdm(
                instrument_df.iterrows(),
                total=len(instrument_df),
                desc=f"Processing mean embeddings for instrument {instrument}"
            ):
                # Encode the audio file using the CLAP model
                timbre_audio_i = clap.encode_audio(os.path.join(audios_path, row["Path"]))
                embeddings.append(timbre_audio_i)

            # Compute the mean embedding across all samples for the instrument
            audio_embedding = torch.mean(torch.stack(embeddings), dim=0)

            # Save the mean embedding to disk
            torch.save(audio_embedding, f"{embeddings_path}{instrument}_embedding.pt")
