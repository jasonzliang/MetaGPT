
def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    elif n == 2:
        return True
    else:
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

def words_in_sentence(sentence):
    """Return a string of words whose lengths are prime numbers."""
    words = sentence.split()  # Split the sentence into words
    primes = [word for word in words if is_prime(len(word))]  # Filter out words with prime length
    return ' '.join(primes)  # Join the words back into a string
