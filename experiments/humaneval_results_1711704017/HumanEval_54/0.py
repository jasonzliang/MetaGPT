
def same_chars(s0: str, s1: str) -> bool:
    """
    Check if two words have the same characters.
    """
    # Sort both strings and compare them
    return sorted(s0) == sorted(s1)

# Test cases
print(same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc'))  # True
print(same_chars('abcd', 'ddaabbc'))  # False
