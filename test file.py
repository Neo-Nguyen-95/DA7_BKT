class Parent:
    def __init__(self, value1, value2):
        self.value1 = value1
        self.value2 = value2
        print(f"Parent initialized with value: {self.value1}")

class Child(Parent):
    def __init__(self, value1, value2, extra):
        # The parent's __init__ is not called automatically.
        # If we don't call it, self.value won't be set.
        super().__init__(value1, value2)
        self.extra = extra
        print(f"Child initialized with extra: {self.extra}")

child = Child(10, 12, "extra data")
# Output:
# Child initialized with extra: extra data

# Trying to access child.value will result in an AttributeError.

child.extra

child.value1

child.value2
