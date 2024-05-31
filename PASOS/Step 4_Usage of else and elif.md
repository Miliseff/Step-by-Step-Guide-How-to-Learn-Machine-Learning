# Conditional Statements in Python 🎛️

A code is essentially a set of instructions executed one after another. Thanks to control structures, we can change the flow of a program, making certain blocks of code execute if and only if specific conditions are met. 🛤️

## Usage of `if` 🔍

An example would be if we have two values `a` and `b` that we want to divide. Before entering the code block that performs `a/b`, it would be important to check that `b` is not zero, as division by zero is undefined. This is where `if` conditionals come in.

```python
a = 4
b = 2
if b != 0:
    print(a / b)
```
In this example, we see how to use an if statement in Python. The != operator checks that b is not zero, and if so, the indented code is executed. Thus, an if statement has two parts:

1-The condition that must be met for the code block to execute, in our case b != 0.
2-The code block that will execute if the condition is met.

It is very important to note that the if statement must end with a : and the code block to execute must be indented. If you use a code editor, indentation will likely occur automatically when you press enter. Note that the code block can contain more than one line, meaning it can contain more than one instruction.

```Python
if b != 0:
    c = a / b
    d = c + 1
    print(d)
```


Everything after the if and indented will be part of the code block that executes if the condition is met. Therefore, the second print() "Outside if" will always execute since it's outside the if block.
```Python
if b != 0:
    c = a / b
    print("Inside if")
print("Outside if")
```


There are other operators that will be covered in other chapters, such as comparing if one number is greater than another. Their use is the same as above.
```Python
if b > 0:
    print(a / b)
```
You can also combine multiple conditions between the if and the :. For example, you can require a number to be greater than 5 and also less than 15. We have three operators used together, which will be evaluated separately until they return the final result, which will be True if the condition is met or False otherwise.
```Python
a = 10
if a > 5 and a < 15:
    print("Greater than 5 and less than 15")
```

It's crucial to note that unlike other languages, in Python, there cannot be an empty if block. The following code would raise a SyntaxError.
```Python
if a > 5:
```

Therefore, if we have an empty if, perhaps because it’s a pending task we’re leaving to implement in the future, we need to use pass to avoid the error. Pass does nothing; it’s just to keep the code interpreter happy.
```Python
if a > 5:
    pass
```
It's not very recommended, but it's possible to put the entire block inside the if on the same line, right after the :. If the code block is not very long, it can be useful to save some lines of code.
```Python
if a > 5: print("It's > 5")
```
If your code block has more than one line, you can also put them on the same line, separating them with ;.
```Python
if a > 5: print("It's > 5"); print("Inside the if")
```

 # Usage of else and elif 🌐

It’s possible that we not only want to do something if a particular condition is met but also want to do something else otherwise. This is where the else clause comes in. The if part behaves as we’ve already explained, with the difference that if the condition is not met, the code inside the else executes. Both code blocks are exclusive; one or the other executes, but never both.
```Python
x = 5
if x == 5:
    print("It's 5")
else:
    print("It's not 5")
```
So far, we’ve seen how to execute a code block if a condition is met or another if not. However, in many cases, we might have several different conditions and want distinct code for each. This is where elif comes into play.
```Python
x = 5
if x == 5:
    print("It's 5")
elif x == 6:
    print("It's 6")
elif x == 7:
    print("It's 7")
```

With the elif clause, we can execute as many different code blocks as we want based on the condition. Translated to natural language, it would be like saying: if it’s equal to 5, do this; if it’s equal to 6, do that; if it’s equal to 7, do something else.

You can also use everything together: if with elif and a final else. It’s important to note that there can be only one if and one else, but there can be multiple elif.
```Python
x = 5
if x == 5:
    print("It's 5")
elif x == 6:
    print("It's 6")
elif x == 7:
    print("It's 7")
else:
    print("It's something else")
```

If you come from other programming languages, you might know that switch is an alternative to elif; however, in Python, this clause does not exist.

 # Ternary Operator 🎭
The ternary operator is a powerful tool that many programming languages have. It’s an if, else clause defined in a single line and can be used, for example, inside a print().
```Python
x = 5
print("It's 5" if x == 5 else "It's not 5")

#Output: It's 5
```
There are three parts to a ternary operator, which are exactly the same as in an if else. We have the condition to evaluate, the code that executes if the condition is met, and the code that executes if it’s not met. In this case, we have all three on the same line.

[code if condition is met] if [condition] else [code if condition is not met]


It’s very useful and can save some lines of code, as well as increase the speed at which we write. For example, if we have a variable to which we want to assign a value based on a condition, it can be done as follows. Continuing with the previous example, in the following code, we try to divide a by b. If b is different from zero, the division is performed and stored in c; otherwise, -1 is stored. That -1 could be a way to indicate there was an error with the division.
```Python
a = 10
b = 5
c = a / b if b != 0 else -1
print(c)
#Output: 2
```

if Examples 📝
Check if a number is even or odd
```Python
x = 6
if not x % 2:
    print("It's even")
else:
    print("It's odd")
```
Decrease `x` by 1 unit if it's greater than zero
```Python
x = 5
x -= 1 if x > 0 else x
print(x)
```
