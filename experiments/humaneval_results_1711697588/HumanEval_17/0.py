
def parse_music(music_string):
    music_dict = {'o': 4, 'o|': 2, '.|': 1}
    return [music_dict[note.strip()] for note in music_string.split(' ')]
    ```
    