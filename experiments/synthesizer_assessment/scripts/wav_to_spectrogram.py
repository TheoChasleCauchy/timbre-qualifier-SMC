import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import ScalarFormatter

def wav_to_spectrogram(wav_path, duration=None, n_fft=2048, hop_length=128, win_length=2048, dpi=300, verbose=False):
    """
    Compute and save spectrograms for two audio files with the same color scale and duration.
    Exports as PDF with rounded y-axis ticks (logarithmic scale).

    Args:
        wav_path (str): Path to the audio file.
        duration (float): Desired duration in seconds for the spectrogram.
        n_fft (int): Number of FFT components.
        hop_length (int): Number of samples between successive frames.
        win_length (int): Window length for STFT.
        dpi (int): DPI for the saved spectrogram image.
        verbose (bool): Whether to print verbose output.
    """

    def load_and_pad_or_trim(y, sr, target_duration):
        target_samples = int(target_duration * sr)
        if len(y) < target_samples:
            # Pad with zeros if too short
            y_padded = np.pad(y, (0, target_samples - len(y)), mode='constant')
        else:
            # Trim if too long
            y_padded = y[:target_samples]
        return y_padded

    # Load and pad/trim both audio files to the same duration
    y1, sr1 = librosa.load(wav_path, sr=None)

    if duration != None:
        y1 = load_and_pad_or_trim(y1, sr1, duration)

    # Compute STFT for both files
    D1 = librosa.stft(y1, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Convert to dB scale
    S_db = librosa.amplitude_to_db(np.abs(D1), ref=np.max)

    plt.figure(figsize=(15, 6), dpi=dpi)
    ax = plt.gca()
    img = librosa.display.specshow(S_db, y_axis='log', x_axis='time', sr=sr1, hop_length=hop_length, ax=ax, cmap="magma_r")
    ax.yaxis.clear()
    ax.set_yticks([100, 250, 500, 1000, 2000, 4000, 8000])  # Rounded y-axis ticks# Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=26)

    ax.yaxis.set_major_formatter(ScalarFormatter())  # Ensure ticks are not in scientific notation

    # Increase axis label size
    ax.set_xlabel('Time (s)', fontsize=30) 
    ax.set_ylabel('Frequency (Hz)', fontsize=30)


    base_name = os.path.splitext(wav_path)[0]
    pdf_path = f"{base_name.replace(' ', '_')}.pdf"
    plt.savefig(pdf_path, format='pdf', dpi=dpi, bbox_inches='tight')
    if verbose:
        print(f"Spectrogram saved to: {pdf_path}")
    plt.close()

def wav_to_spectrogram_pair(wav_path1, wav_path2, duration, n_fft=2048, hop_length=128, win_length=2048, dpi=300, verbose=False):
    """
    Compute and save spectrograms for two audio files with the same color scale and duration.
    Exports as PDF with rounded y-axis ticks (logarithmic scale).

    Args:
        wav_path1 (str): Path to the first audio file.
        wav_path2 (str): Path to the second audio file.
        duration (float): Desired duration in seconds for both spectrograms.
        n_fft (int): Number of FFT components.
        hop_length (int): Number of samples between successive frames.
        win_length (int): Window length for STFT.
        dpi (int): DPI for the saved spectrogram image.
    """

    def load_and_pad_or_trim(y, sr, target_duration):
        target_samples = int(target_duration * sr)
        if len(y) < target_samples:
            # Pad with zeros if too short
            y_padded = np.pad(y, (0, target_samples - len(y)), mode='constant')
        else:
            # Trim if too long
            y_padded = y[:target_samples]
        return y_padded

    # Load and pad/trim both audio files to the same duration
    y1, sr1 = librosa.load(wav_path1, sr=None)
    y2, sr2 = librosa.load(wav_path2, sr=None)

    # Ensure both files have the same sample rate
    if sr1 != sr2:
        raise ValueError("Sample rates of the two audio files must match.")

    y1 = load_and_pad_or_trim(y1, sr1, duration)
    y2 = load_and_pad_or_trim(y2, sr2, duration)

    # Compute STFT for both files
    D1 = librosa.stft(y1, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    D2 = librosa.stft(y2, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Convert to dB scale
    S_db1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
    S_db2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)

    # Find global min/max for consistent color scale
    global_min = min(S_db1.min(), S_db2.min())
    global_max = max(S_db1.max(), S_db2.max())

    # Plot both spectrograms with the same color scale
    for i, (S_db, wav_path) in enumerate(zip([S_db1, S_db2], [wav_path1, wav_path2]), 1):
        plt.figure(figsize=(15, 6), dpi=dpi)
        ax = plt.gca()
        img = librosa.display.specshow(S_db, y_axis='log', x_axis='time', sr=sr1, hop_length=hop_length,
                                        vmin=global_min, vmax=global_max, ax=ax, cmap="magma_r")
        ax.yaxis.clear()
        ax.set_yticks([100, 250, 500, 1000, 2000, 4000, 8000])  # Rounded y-axis ticks# Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=26)

        ax.yaxis.set_major_formatter(ScalarFormatter())  # Ensure ticks are not in scientific notation

        # Increase axis label size
        ax.set_xlabel('Time (s)', fontsize=30) 
        ax.set_ylabel('Frequency (Hz)', fontsize=30)


        base_name = os.path.splitext(wav_path)[0]
        pdf_path = f"{base_name.replace(' ', '_')}.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"Spectrogram {i} saved to: {pdf_path}")
        plt.close()

# Example usage:
# wav_to_spectrogram_pair(
#     "path/WOOD_BLOCK/WOOD BLOCK_ground_truth_furthest_neighbor_trait_percussive_dist_0.61_sample_95_note_E4.wav",
#     "path/WOOD_BLOCK/WOOD BLOCK_ground_truth_nearest_neighbor_trait_percussive_dist_0.01_411WBNO3_03.wav",
#     duration = 1.2
#     )
# wav_to_spectrogram_pair(
#     "path/CELLO/CELLO_ground_truth_furthest_neighbor_trait_resonant-vibrant_dist_0.22_sample_35_note_G3.wav",
#     "path/CELLO/CELLO_ground_truth_nearest_neighbor_trait_resonant-vibrant_dist_0.03_171VCNOF_64.wav",
#     duration = 2.6
#     )

