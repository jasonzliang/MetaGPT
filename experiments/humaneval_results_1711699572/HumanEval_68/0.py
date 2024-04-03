
def pluck(arr):
    even_values = [i for i in arr if i % 2 == 0]
    if not even_values:
        return []
    smallest_even_value = min(even_values)
    index = arr.index(smallest_even_value)
    return [smallest_even_value, index]
