
    def find_closest_elements(numbers):
        numbers.sort()
        closest = (numbers[0], numbers[1])
        min_difference = abs(numbers[0] - numbers[1])
        
        for i in range(1, len(numbers) - 1):
            difference = abs(numbers[i] - numbers[i+1])
            
            if difference < min_difference:
                closest = (numbers[i], numbers[i+1])
                min_difference = difference
                
        return closest
    