
def by_length(arr):
    nums = [i for i in arr if 1 <= i <= 9]
    nums.sort()
    nums.reverse()
    names = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    return [names[i-1] for i in nums]
