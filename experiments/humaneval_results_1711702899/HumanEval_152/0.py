
def compare_scores(game, guess):
    """Compares the scores of a game to the guesses of a player.

    Args:
        game: A list of integers representing the actual scores of each match in the game.
        guess: A list of integers representing the guessed scores of each match by the player.

    Returns:
        A list of integers where each index represents how far off the corresponding guess was from the score. 
        If the guess is correct, the value at that index will be 0.

    Raises:
        ValueError: If game and guess are not lists or if they have different lengths.
    """
    
    # Checking if both inputs are lists
    if not (isinstance(game, list) and isinstance(guess, list)):
        raise ValueError("Both game and guess must be of type list.")
        
    # Checking if both lists have the same length
    if len(game) != len(guess):
        raise ValueError("Game and guess must have the same number of elements.")
    
    return [abs(g - gs) for g, gs in zip(game, guess)]
