---- Python Syntax 🐍----

The term syntax refers to the set of rules that define how code should be written in a particular programming language.

Syntax is to programming what grammar is to languages.

Python does not support the use of `$` nor is it necessary to end lines with `;` like in other languages, and there's no need to use `{}` for control structures like `if`.

if ($variable){
    x=9;
}

Let's see an example where we use strings, arithmetic operators, and the `if` conditional.

The following code simply defines three values `a`, `b`, and `c`, performs some operations with them, and displays the result on the screen.

# Define a variable x with a string
x = "The value of (a+b)*c is"

# We can perform multiple assignments
a, b, c = 4, 3, 2

# Perform operations with a, b, c
d = (a + b) * c

# Define a boolean variable
to_print = True

# If to_print, print()
if to_print:
    print(x, d)

# Output: The value of (a+b)*c is 14

As you can see, Python syntax is very similar to natural language or pseudocode, making it relatively easy to read. Another advantage is that we don't need anything else; the above code can be executed as is. If you know other languages like C or Java, you'll find this comfortable, as there's no need to create a typical `main()` function.



------ Commenting 🗒️-----

Comments are blocks of text used to explain the code. They provide relevant information to other programmers or our future selves about the written code.

Comments start with `#`, and everything after it on the same line is considered a comment.

# This is a comment

Like in other programming languages, we can also comment multiple lines of code. To do this, we use triple quotes, either single `'''` or double `"""`, to open and close the comment block.

'''
This is a comment
spanning multiple lines
of code
'''



----Indentation and Code Blocks 🧱----

In Python, code blocks are represented by indentation. Although there's some debate about using tabs or spaces, the general rule is to use four spaces.

In the following code, we have an `if` conditional. Right after it, we have a `print()` indented with four spaces. Therefore, everything with that indentation belongs to the `if` block.

if True:
    print("True")

This is very important because the following code is not the same. In fact, the next code would give an error since the `if` statement doesn't contain any code block, which is something you can't do in Python.

if True:
print("True")

Unlike other programming languages, it's not necessary to use `;` to end each line in Python.

# Other languages like C
# require ; at the end of each line

x = 10;

However, in Python, it's not necessary; just a line break is enough.

x = 5
y = 10

But you can use a semicolon `;` to have two statements on the same line.

x = 5; y = 10



---- Multiple Lines 📜----

In some situations, you might want to have a single instruction span multiple lines of code. One of the main reasons could be that it's too long, and in fact, in the PEP8 specification, it's recommended that lines not exceed 79 characters.

Using `\`, you can break the code into multiple lines, which in certain cases makes the code much more readable.

x = 1 + 2 + 3 + 4 +\
    5 + 6 + 7 + 8

On the other hand, if we're within a block surrounded by parentheses `()`, just jump to the next line.

x = (1 + 2 + 3 + 4 +
     5 + 6 + 7 + 8)

The same can be done for function calls.

def my_function(a, b, c):
    return a + b + c

d = my_function(10,
                23,
                3)



---- Creating Variables 📊----

You can create a variable and assign a value to it using `=`, but there are also other ways to do it in a slightly more sophisticated manner.

For example, we can assign the same value to different variables with the following code.

x = y = z = 10

Or we can assign multiple values separated by commas.

x, y = 4, 2
x, y, z = 1, 2, 3



---- Naming Variables 📝----

You can name your variables however you like, but it's important to know that uppercase and lowercase letters are distinct. Variables `x` and `X` are different.

Additionally, there are certain rules when naming variables:

- The name cannot start with a number.
- Hyphens (`-`) are not allowed.
- Spaces are not allowed.

Here are examples of valid and invalid variable names:

# Valid
_variable = 10
vari_able = 20
variable10 = 30
variable = 60
variaBle = 10

# Invalid
2variable = 10
var-iable = 10
var iable = 10

Another condition for naming a variable in Python is not to use reserved Python names. Reserved words are used internally by Python, so we cannot use them for our variables or functions.

import keyword
print(keyword.kwlist)

# Output:
# ['False', 'None', 'True', 'and', 'as', 'assert',
# 'async', 'await', 'break', 'class', 'continue',
# 'def', 'del', 'elif', 'else', 'except', 'finally',
# 'for', 'from', 'global', 'if', 'import', 'in', 'is',
# 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise',
# 'return', 'try', 'while', 'with', 'yield']

In fact, with the following command, you can see all the keywords that cannot be used:

import keyword
print(keyword.kwlist)

----Use of Parentheses 🎯----

Python supports all common mathematical operators, known as arithmetic operators. Therefore, we can perform additions, subtractions, multiplications, exponents (using `**`), and others that we won't explain right now. In the following example, we perform several operations on the same line and store the result in `y`.

x = 10
y = x*3-3**10-2+3

However, the behavior of the above code and the following one is different because the use of parentheses `()` prioritizes certain operations over others.

x = 10
y = (x*3-3)**(10-2)+3

Parentheses are not only applied to arithmetic operators but can also be used with other operators like relational or membership operators that we see in other posts.

---- Variables and Scope 🔍----

A very important concept when defining a variable is to know its scope. In the following example, the variable with the value `10` has a global scope, and the one with the value `5` inside the function has a local scope. This means that when we do `print(x)`, we are accessing the global variable `x`, not the `x` defined within the function.

x = 10

def my_function():
    x = 5

my_function()
print(x)

Finally, in any programming language, it's important to understand what's happening as different instructions are executed. Therefore, it's interesting to use `print()` in different sections of the code as it allows us to see the value of variables and other useful information.

There are many ways to use the `print()` function, and we explain them in detail in this post, but for now, it's enough for you to know the basics.

As we've already seen, you can use `print()` to print any text you want to the screen.

print("This is the content to print")

It's also possible to print the content of a variable.

x = 10
print(x)

And by separating values with commas `,`, you can print text and the content of variables.

x = 10
y = 20
print("The values of x, y are:", x, y)

# Output: The values of x, y are: 10 20
