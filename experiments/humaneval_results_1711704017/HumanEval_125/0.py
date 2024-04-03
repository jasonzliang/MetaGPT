
def split_words(txt):
    """
    Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
    should split on commas ',' if no commas exists you should return the number of lower-case letters with odd order in the
    alphabet, ord('a') =0, ord('b') = 1, ... ord('z') = 25.
    """
    
    # Split words by whitespace
    split_by_space = txt.split()
    if len(split_by_space) > 1:
        return split_by_space
    
    # If no whitespaces, split by commas
    split_by_comma = txt.split(',')
    if len(split_by_comma) > 1:
        return split_by_comma
    
    # Count lower-case letters with odd order in the alphabet
    count = 0
    for char in txt:
        if char.islower() and (ord(char) - ord('a')) % 2 == 1:
            count += 1
            
    return count
```

# Example usage:
print(split_words("Hello world!")) # Outputs: ["Hello", "world!"]
print(split_words("Hello,world!")) # Outputs: ["Hello", "world!"]
print(split_words("abcdef")) # Outputs: 3
