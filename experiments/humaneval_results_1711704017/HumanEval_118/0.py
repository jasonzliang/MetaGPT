
def get_closest_vowel(word):
    """Find the closest vowel that stands between two consonants from the right side of the word."""
    vowels = 'aeiouAEIOU'
    for i in range(len(word) - 1, 0, -1):
        if word[i] not in vowels and word[i-1] in vowels:
            return word[i-1]
    return ''
