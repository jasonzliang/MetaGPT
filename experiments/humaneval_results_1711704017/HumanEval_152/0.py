
def compare_scores(game, guess):
    """Compare the scores of a game with the guesses. 
    Return an array of differences between the score and the guess."""
    if len(game) != len(guess):
        raise ValueError("The length of game and guess arrays should be equal.")
    
    diff = []
    for i in range(len(game)):
        # If the guess is correct, append 0 to the differences list.
        if game[i] == guess[i]:
            diff.append(0)
        else:
            # Otherwise, append the absolute difference between the score and the guess.
            diff.append(abs(game[i]-guess[i]))
    return diff
