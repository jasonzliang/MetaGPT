
def select_words(s, n):
    vowels = 'aeiou'
    words = s.split()
    result = []
    for word in words:
        if sum([1 for c in word.lower() if c not in vowels]) == n:
            result.append(word)
    return result
