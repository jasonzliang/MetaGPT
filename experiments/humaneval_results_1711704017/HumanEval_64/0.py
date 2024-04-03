
def vowels_count(s):
    """Write a function vowels_count which takes a string representing
    a word as input and returns the number of vowels in the string.
    Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a
    vowel, but only when it is at the end of the given word."""
    
    # Ensure input is a string
    if not isinstance(s, str):
        raise ValueError("Input must be a string")
        
    s = s.lower()  # Convert to lowercase for case-insensitive comparison
    vowels = 'aeiou'
    
    count = 0
    for char in s:
        if char in vowels:
            count += 1
            
    # If the last character is 'y', increment the count
    if s[-1] == 'y':
        count += 1
        
    return count
