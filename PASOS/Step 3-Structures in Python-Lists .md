# Data Types in Python 🔢

## Integers (int) ➕➖

Integers in Python allow us to store non-decimal numeric values, either positive or negative, of any value. The `type()` 
function returns the type of the variable, and we can see that it is indeed of the class `int`

'''
i = 12
print(i)          # 12
print(type(i))    # <class 'int'>
'''

### Convert to int 🔄

It is possible to convert another type to `int`.
As explained, the `int` type cannot contain decimals, so if we try to convert a decimal number, everything to the right of the decimal point will be truncated.

'''
b = int(1.6)
print(b) # 1
'''

## Booleans (bool) ✔️❌

### Declare Boolean Variables
Boolean variables can be declared as follows:

'''
x = True
y = False
'''

### Evaluate Expressions

A boolean value can also be the result of evaluating an expression. Certain operators like greater than, less than, or equal to return a boolean value.
'''
print(1 > 0)  # True
print(1 <= 0) # False
print(9 == 9) # True
'''
### `bool` Function

'''It is also possible to convert a certain value to `bool` using the `bool()` function.'''

print(bool(10))     # True
print(bool(-10))    # True
print(bool("Hola")) # True
print(bool(0.1))    # True
print(bool([]))     # False


### Usage with `if`

'''Conditional statements `if` evaluate a condition that is a boolean value.'''

a = 1
b = 2
if b > a:
    print("b is greater than a")


'''The expression after the `if` is always evaluated until it yields a boolean.'''

if True:
    print("It's True")


## Floating Point Numbers (float) 🌊

'''The numeric type `float` allows representing a positive or negative number with decimals, i.e., real numbers. 
If you come from other languages, you might know the double type, which means it has double the precision of a float.
In Python, things are a bit different, and floats are actually double.

If we declare a variable and assign it a decimal value, by default the variable will be of type `float`.'''

f = 0.10093
print(f)       # 0.10093
print(type(f)) # <class 'float'>

### Convert to float 🔄

'''It can also be declared using scientific notation with `e` and the exponent. 
The following example would be the same as saying 1.93 multiplied by ten raised to the power of -3.'''

f = 1.93e-3

'''We can also convert another type to `float` using `float()`. We can see how `True` is actually treated as 1, and when converted to `float`, it becomes 1.0.'''

a = float(True)
b = float(1)
print(a, type(a)) # 1.0 <class 'float'>
print(b, type(b)) # 1.0 <class 'float'>

### Representable Range

'''One curiosity is that floats do not have infinite precision. 
We can see in the following example how `f` is actually stored as 1, since it is not possible to represent such decimal precision.'''


f = 0.99999999999999999
print(f)      # 1.0
print(1 == f) # True


'''Floats, unlike `int`, have minimum and maximum values they can represent. 
The minimum precision is `2.2250738585072014e-308` and the maximum is `1.7976931348623157e+308`, but if you don't believe me, you can verify it yourself.
'''

import sys
print(sys.float_info.min) # 2.2250738585072014e-308
print(sys.float_info.max) # 1.7976931348623157e+308


'''In fact, if you try to assign a value greater than the maximum to a variable, the variable takes the value `inf` or infinity.'''

f = 1.7976931348623157e+310
print(f) # inf


'''If you want to represent a variable with a very high value, you can also directly create a variable that contains that `inf` value.'''

f = float('inf')
print(f) # inf


## Strings in Python 🎼

'''Strings in Python are an immutable type that allows storing sequences of characters. To create one, you need to enclose the text in double quotes `"`. You can get more help with the command `help(str)`.
'''

s = "This is a string"
print(s)       # This is a string
print(type(s)) # <class 'str'>


'''It is also valid to declare strings with single quotes `'`.'''

s = 'This is another string'
print(s)        # This is another string
print(type(s))  # <class 'str'>


'''Strings are not limited in size, so the only limit is your computer's memory. A string can also be empty.'''

s = ''

'''A situation that often occurs is when we want to include a quote, either single `'` or double `"`, within a string. 
If we do it the following way, we would have an error, since Python doesn't know very well where it starts and ends.'''

# s = "This is a double quote " example" # Error!


'''To solve this problem, we must resort to escape sequences. 
In Python, there are several, but we will analyze them in more detail in another chapter. For now, the most important is `\"`, which allows us to embed quotes within a string.'''

s = "This is a double quote \" example"
print(s) # This is a double quote " example


'''We can also include a newline within a string, which means that what is after the newline will be printed on a new line.'''

s = "First line\nSecond line"
print(s)
# First line
# Second line

'''We can also use `\` followed by a number, which will print the associated character. In this case, we print character 110, which corresponds to `H`.'''

print("\110\110") # HH


### String Formatting 🎨

'''We might want to declare a string that contains variables within it, such as numbers or even other strings. 
One way to do this is by concatenating the string we want with another using the `+` operator. Note that `str()` converts to string what is passed as a parameter.'''

x = 5
s = "The number is: " + str(x)
print(s) # The number is: 5


'''Another way is using `%`. 
On one side we have `%s` indicating the type we want to print, and on the other, to the right of the `%`, we have the variable to print. 
To print a string, we would use `%s` or `%f` for a floating-point value.'''


x = 5
s = "The number is: %d" % x
print(s) # The number is: 5


'''If we have more than one variable, we can also do it by passing the parameters within `()`. 
If you come from languages like C, this way will be very familiar to you. However, this is not the preferred way to do it now that we have newer versions of Python.'''


s = "The numbers are %d and %d." % (5, 10)
print(s) # The numbers are 5 and 10.


'''A slightly more modern way to do the same is by using `format()`.'''

s = "The numbers are {} and {}".format(5, 10)
print(s) # The numbers are 5 and 10.


'''It is also possible to name each element, and `format()` will take care of replacing everything.'''


s = "The numbers are {a} and {b}".format(a=5, b=10)
print(s) # The numbers are 5 and 10.


'''As if there weren't enough already, there is a third way to do it introduced in Python version 3.6. 
They are called f-strings or formatted string literals. This new feature allows embedding expressions within strings.'''


a = 5; b = 10
s = f"The numbers are {a} and {b}"
print(s) # The numbers are 5 and 10.


'''You can even perform operations within the creation of the string.'''

a = 5; b = 10
s = f"a + b = {a+b}"
print(s) # a + b = 15


'''You can even call a function within.'''


def function():
    return 20
s = f"The function result is {function()}"
print(s) # The function result is 20


## Lists in Python 📜

'''Lists in Python are a data type that allows storing data of any
type in a sequence. To declare a list, we place the elements within `[]` and separate them by commas.'''

l = [1, 2, 3, 4]
print(l)       # [1, 2, 3, 4]
print(type(l)) # <class 'list'>


'''The elements of a list do not necessarily have to be of the same type. You can have integers, floats, strings, and other lists.'''


l = [1, 2.2, "Hola", [1, 2, 3]]


'''In this example, we have created a list that contains an integer, a float, a string, and a sublist of three integers. 
To access an element within the list, we use `[]` and specify the index of the element we want to retrieve.'''


l = [1, 2, 3, 4]
print(l[0]) # 1
print(l[1]) # 2
print(l[2]) # 3
print(l[3]) # 4


'''Remember that in Python, indices start at `0`, not `1`. In the previous example, to get the first element, we have to use index `0`.'''

### Negative Indices ➖

'''We can also use negative indices to access elements from the end of the list.'''


l = [1, 2, 3, 4]
print(l[-1]) # 4
print(l[-2]) # 3


'''In this case, the first element from the end is `4` and the second element from the end is `3`.'''

### List Operations 🔧

'''Lists can be modified by adding, deleting, or changing elements.'''

#### Add Element ➕

'''To add an element to a list, we use the `append()` method.'''


l = [1, 2, 3]
l.append(4)
print(l) # [1, 2, 3, 4]


'''If we want to add an element at a specific position, we use the `insert()` method.'''


l = [1, 2, 3]
l.insert(1, 1.5)
print(l) # [1, 1.5, 2, 3]


'''In this case, we have inserted `1.5` in the position `1`.'''

#### Delete Element ➖

'''To delete an element from a list, we use the `remove()` method. If the element is not found, it raises a `ValueError`.'''


l = [1, 2, 3]
l.remove(2)
print(l) # [1, 3]


'''Another way to delete an element is using `del` followed by the index of the element to delete.'''


l = [1, 2, 3]
del l[1]
print(l) # [1, 3]


#### Change Element 🔄

'''To change an element in a list, we specify the index and assign the new value.'''


l = [1, 2, 3]
l[1] = 4
print(l) # [1, 4, 3]


## Dictionaries in Python 📚

'''Dictionaries in Python are collections of key-value pairs. Each key is associated with a value, and you can use the key to retrieve the value.
Dictionaries are created using curly braces `{}`.'''


d = {'name': 'John', 'age': 30, 'city': 'New York'}
print(d)       # {'name': 'John', 'age': 30, 'city': 'New York'}
print(type(d)) # <class 'dict'>


### Access Value 🔍

'''To access a value in a dictionary, we use the key inside square brackets `[]`.'''


print(d['name']) # John


'''If the key does not exist, it raises a `KeyError`.'''

### Add or Change Key-Value Pair ➕

'''To add or change a key-value pair, we use the key inside square brackets and assign the new value.'''


d['email'] = 'john@example.com'
print(d) # {'name': 'John', 'age': 30, 'city': 'New York', 'email': 'john@example.com'}


### Delete Key-Value Pair ➖

'''To delete a key-value pair, we use the `del` keyword followed by the key inside square brackets.'''


del d['age']
print(d) # {'name': 'John', 'city': 'New York', 'email': 'john@example.com'}


### Dictionary Methods 🔧

'''Dictionaries have several useful methods.'''

#### `keys()`

'''The `keys()` method returns a view object that displays a list of all the keys in the dictionary.'''


print(d.keys()) # dict_keys(['name', 'city', 'email'])


#### `values()`

'''The `values()` method returns a view object that displays a list of all the values in the dictionary.'''

print(d.values()) # dict_values(['John', 'New York', 'john@example.com'])


#### `items()`

'''The `items()` method returns a view object that displays a list of key-value pairs as tuples.'''

print(d.items()) # dict_items([('name', 'John'), ('city', 'New York'), ('email', 'john@example.com')])


### Check if Key Exists 🔍

'''To check if a key exists in a dictionary, we use the `in` keyword.'''

if 'name' in d:
    print("Name exists")
else:
    print("Name does not exist")


## Tuples in Python 🔗

'''Tuples are similar to lists, but they are immutable. This means that once a tuple is created, you cannot change its elements. 
Tuples are created using parentheses `()`.'''


t = (1, 2, 3)
print(t)       # (1, 2, 3)
print(type(t)) # <class 'tuple'>


### Access Element 🔍

'''To access an element in a tuple, we use the index inside square brackets `[]`.'''

print(t[1]) # 2

### Immutable Nature 🔒

'''Since tuples are immutable, you cannot change, add, or remove elements.'''


# t[1] = 4 # TypeError: 'tuple' object does not support item assignment


### Tuple Methods 🔧

'''Tuples have only two methods: `count()` and `index()`.'''

#### `count()`

'''The `count()` method returns the number of times a specified value appears in the tuple.'''

print(t.count(2)) # 1


#### `index()`

'''The `index()` method returns the index of the first occurrence of a specified value.'''

print(t.index(3)) # 2


### Unpack Tuple 📦

'''You can unpack a tuple into variables.'''

a, b, c = t
print(a, b, c) # 1 2 3


### Single Element Tuple ☝️

'''To create a tuple with a single element, you need to include a comma `,` after the element.'''

t = (1,)
print(t)       # (1,)
print(type(t)) # <class 'tuple'>


## Sets in Python 🔥

'''Sets are collections of unique elements. Sets are created using curly braces `{}` or the `set()` function.'''


s = {1, 2, 3}
print(s)       # {1, 2, 3}
print(type(s)) # <class 'set'>


### Add Element ➕

'''To add an element to a set, we use the `add()` method.'''

s.add(4)
print(s) # {1, 2, 3, 4}


### Remove Element ➖

'''To remove an element from a set, we use the `remove()` method. If the element does not exist, it raises a `KeyError`'''

s.remove(3)
print(s) # {1, 2, 4}


### Set Operations 🔧

'''Sets support mathematical operations like union, intersection, and difference.'''

#### Union ∪

'''The union of two sets is a set containing all the elements of both sets. We use the `union()` method or the `|` operator.'''


s1 = {1, 2, 3}
s2 = {3, 4, 5}
print(s1.union(s2)) # {1, 2, 3, 4, 5}
print(s1 | s2)      # {1, 2, 3, 4, 5}


#### Intersection ∩

'''The intersection of two sets is a set containing the elements common to both sets. We use the `intersection()` method or the `&` operator.'''

print(s1.intersection(s2)) # {3}
print(s1 & s2)             # {3}


#### Difference −

'''The difference of two sets is a set containing the elements of the first set that are not in the second set. We use the `difference()` method or the `-` operator.'''

print(s1.difference(s2)) # {1, 2}
print(s1 - s2)           # {1, 2}


### Check if Element Exists 🔍

'''To check if an element exists in a set, we use
the `in` keyword.'''

if 1 in s:
    print("1 exists in the set")
else:
    print("1 does not exist in the set")


## Conclusion 🏁

'''In this tutorial, we have explored the four main collection types in Python: lists, dictionaries, tuples, and sets.
Each of these types has its own characteristics and methods. Lists and dictionaries are mutable, while tuples are immutable. Sets are collections of unique elements and support mathematical operations.

Understanding these collection types is essential for effective programming in Python. 
They allow us to store, manage, and manipulate data in various ways. With this knowledge, you are now equipped to use these powerful tools in your Python projects.
'''
