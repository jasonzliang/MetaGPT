
def compare(game, guess):
    return [abs(g - s) if g != s else 0 for g, s in zip(guess, game)]
    