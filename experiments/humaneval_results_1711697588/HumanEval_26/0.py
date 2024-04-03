
def remove_duplicates(numbers):
    result = []
    for number in numbers:
        if number not in result:
            result.append(number)
    return result
