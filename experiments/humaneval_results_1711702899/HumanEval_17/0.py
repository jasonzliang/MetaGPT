
from typing import List

def parse_music(music_string: str) -> List[int]:
    """Parses a string representing musical notes in a special ASCII format 
    and returns list of integers corresponding to how many beats does each note last.

    Args:
        music_string: A string representing musical notes in the following format:
            'o' - whole note, lasts four beats
            'o|' - half note, lasts two beats
            '.|' - quater note, lasts one beat

    Returns:
        List of integers corresponding to how many beats does each note last.

    Raises:
        ValueError: If the input string contains a character that is not 'o', 'o|' or '.|'.
    """
    
    # Define a dictionary mapping notes to their lengths in beats
    note_lengths = {'o': 4, 'o|': 2, '.|': 1}

    try:
        # Split the string into individual notes and map each one to its length in beats
        return [note_lengths[note.strip()] for note in music_string.split(' ')]
    except KeyError as e:
        raise ValueError(f"Invalid note: {e}") from None
