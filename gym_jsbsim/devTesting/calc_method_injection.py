import numpy as np
import types

class Test:
    def __init__(self, calculate_injection = None):
        self.eins = 1
        self.zwei = 2
        self.sieben = 7

        self.arr = np.array(range(10))

        if calculate_injection:
            self.calculate = calculate_injection.__get__(self)
        
    def calculate(self, a):
        print("I am the original")
        return self.zwei * self.arr - self.zwei

if __name__ == "__main__":
    a = Test()
    a.calculate(4)

    def calc(obj, a):
        print("I am the fake")
        return a*obj.arr
    
    b = Test(calculate_injection=calc)
    b.calculate(3)
