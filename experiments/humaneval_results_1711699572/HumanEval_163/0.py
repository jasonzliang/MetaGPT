
def generate_integers(a, b):
    if a > b:
        a, b = b, a
        
    return sorted([i for i in range(a, b + 1) if i % 2 == 0])
