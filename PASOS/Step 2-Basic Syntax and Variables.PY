'''🌍 Hello World in Python

In any introduction to a new programming language, the famous Hello World cannot be missing.'''

print("Hello World")

'''Therefore, you can already imagine that the print() function is used to print values to the screen. It will print everything inside the parentheses.'''

###############################

'''🔢 Variables in Python'''

'''Let's create a variable to store a number. Unlike other programming languages, you don't need to tell Python the data type we want to store in x. In other languages, 
you need to specify that x will store an integer, but not in this case.'''

x = 5

'''Now we can combine the print() function we've seen with the x we've defined, so instead of printing "Hello World", we print the value of x.'''

print(x)
# Output: 5

'''The use of # is Python's way of creating comments.

Now let's add two variables together and print their value. First, we'll declare them with names a and b. Declaring a variable means "creating it".'''

# Declare variables a, b
# and assign two values
a = 3
b = 7

'''Now Python knows a and b and their respective values. We can use + to add them together, and once again use print() to display their value on the screen.'''

print(a + b)

'''It's important to only use variables that have been defined, otherwise, you'll encounter an error. If we do:'''

 ##print(z)
# Error! Variable does not exist

'''We'll get an error because Python doesn't know what z is since it hasn't been declared previously.'''

#####################
'''🔍 A Conditional Statement

The following code uses if to check if a is equal == to 10. If it is, it will print "Es 10"; otherwise, it will print "No es 10". 
The use of ==, which is the relational operator I'll show you more about later, means 'equal'. '''

a = 10
if a == 10:
    print("Es 10")
else:
    print("No es 10")

###################
'''Decimals and Strings

Just like we've seen how a variable can store an integer value like 10, it's also possible to store other types such as decimals or even text strings.

If we want to store a decimal value, simply indicate it using a . as the decimal separator.'''

decimal_value = 10.3234

'''And if we want to store a string, we need to enclose its content in single ' or double " quotes.'''

my_string = "Hola Mundo"

