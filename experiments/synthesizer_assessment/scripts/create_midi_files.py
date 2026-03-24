import mido  # MIDI file manipulation library
from mido import Message, MidiFile, MidiTrack  # MIDI message, file, and track classes
import os  # Operating system interfaces for directory and file operations

def create_midi_files():
    """
    Generate MIDI files for all 128 MIDI notes.

    This function:
    1. Iterates over all 128 MIDI notes.
    2. For each note, creates a MIDI file containing a single note event.
    3. Sets the tempo to 120 BPM and the note duration to 1 second.
    4. Saves each MIDI file to disk with a filename indicating the note number.

    Steps:
    - For each MIDI note (0 to 127), create a new MIDI file.
    - Set the tempo to 120 BPM.
    - Add a note-on event at the start of the track.
    - Add a note-off event after 1 second.
    - Save the MIDI file to disk.

    Returns:
        None: MIDI files are saved to disk in the specified directory structure.
    """
    # Iterate over all 128 MIDI notes
    for note in range(0, 128):
        # Create a new MIDI file and track
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        # Set the tempo to 120 BPM
        tempo = mido.bpm2tempo(120)  # Convert 120 BPM to microseconds per beat
        ticks_per_beat = mid.ticks_per_beat  # Get the ticks per beat from the MIDI file

        # Add a tempo meta message to the track
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))

        # Calculate the duration of the note in ticks
        seconds = 1.0  # Desired note duration in seconds
        beats_per_second = 120 / 60  # 2 beats per second at 120 BPM
        beats = seconds * beats_per_second  # Calculate the number of beats for the duration
        ticks = int(beats * ticks_per_beat)  # Convert beats to ticks

        # Add a note-on message for the current note with maximum velocity
        track.append(Message('note_on', note=note, velocity=100, time=0))

        # Add a note-off message for the current note after the calculated duration
        track.append(Message('note_off', note=note, velocity=0, time=ticks))

        # Create the output directory if it doesn't exist
        os.makedirs("data/TokenSynth/midi_files", exist_ok=True)

        # Save the MIDI file with a filename indicating the note number
        mid.save(f"data/TokenSynth/midi_files/input_midi_{note}.mid")
