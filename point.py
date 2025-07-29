class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __str__(self):
        return "Point: (" + str(self.x) + ", " + str(self.y) + ")"

    def __hash__(self):
        return hash((self.x,self.y))

    def __eq__(self, other):
        if not other:
            return False
        return (self.x,self.y) == (other.x, other.y)

    def __ne__(self, other):
        if not other:
            return True
        return self.x != other.x or self.y != other.y
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
