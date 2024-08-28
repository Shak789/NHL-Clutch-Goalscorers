class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        solution = []
        for i in range(0, n + 1):
            if ((i + 1) % 3 == 0 and (i + 1) % 5 == 0 ):
                soution[i] = "FizzBuzz"
            elif ((i + 1) % 3 == 0):
                soution[i] = "Fizz"
            elif ((i + 1) % 5 == 0):
                solution[i] = "Buzz"
            else:
                string = str(i + 1)
                solution[i] = string
        return solution
                


        