'''Here's a curiosity: unlike in other languages, the function accepts an input parameter without specifying its type.
The variable x first stores a string, then a float, and then an integer. The function() is called with an integer, but its value is divided by 2, 
and the result is automatically converted into a float.
'''
def function(input):
    return input / 2

x = "Hola"
x = 7.0
x = int(x)
x = function(x)
print(x)
print(type(x))


<class 'float'>
