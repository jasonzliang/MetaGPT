
def parse_music(music_string: str) -> List[int]:
    """ Input to this function is a string representing musical notes in a special ASCII format.
    Your task is to parse this string and return list of integers corresponding to how many beats does each
    not last.

    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quater note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """
    # Define a dictionary to map notes to their corresponding beat lengths
    note_to_beat = {'o': 4, 'o|': 2, '.|': 1}
    
    # Split the music string into individual notes and convert them to their corresponding beats
    return [note_to_beat[note] for note in music_string.split()]
