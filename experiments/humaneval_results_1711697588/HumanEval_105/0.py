
def by_length(arr):
    nums = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    return [nums[i] for i in sorted([n for n in arr if 1 <= n <= 9])[::-1]]
    ```
    