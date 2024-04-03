
def is_bored(S):
    """
    Count the number of boredoms in a string. A boredom is a sentence that starts with the word "I".
    Sentences are delimited by '.', '?' or '!'.
    """
    # Initialize count to 0
    count = 0
    
    # Split the string into sentences
    sentences = [s.strip() for s in re.split('(?<=[.!?]) +', S) if s]
    
    # Iterate over each sentence
    for sentence in sentences:
        # If the sentence starts with "I", increment count
        if sentence.startswith("I"):
            count += 1
            
    return count
