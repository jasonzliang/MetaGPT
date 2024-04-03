
def vowels_count(s):
    count = 0
    for char in s.lower():
        if char in 'aeiou':
            count += 1
        elif char == 'y' and len(s) > 1 and s[-2].lower() not in 'aeiou':
            count += 1
    return count
