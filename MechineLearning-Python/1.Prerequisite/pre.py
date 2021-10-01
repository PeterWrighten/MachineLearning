import numpy as np


# print

a = 100
b = 200

print(a + b)

# Calculate

a = b * 3

c = a / 10

print(a, c)

# character

s = 'python'

print(s)

s_1 = s + " programming"

print(s_1)

# Function


def multi_div(x, y):
    return x * y, x / y


m, d = multi_div(20, 5)

print(m, d)

# list

ll = [1, 2, 3, 4, 5]

print(ll)

L = len(ll)

print(L)

print(ll[0], ll[-1], ll[1:3], s_1[2:-3])

# ll[0] is the first, ll[-1] is the last.

# list comprehension

l2 = [v*2 for v in ll]

print(l2)

# More examples about list comprehension

F = ['sb', 'wb', 'disney', 'pixel']

FF = [v for v in F if 'b' in v]

FF_1 = [v for v in F if 'b' not in v]

print(FF, FF_1)

# for loop

for idx in range(len(s)):
    print(idx)

for item in s:
    print(item)

for idx, item in enumerate(s):
    print(idx, item)

# if syntax

if ll[0] == 1:
    print("ll[0] is 1")
elif ll[0] == 0:
    print("ll[0] is 0")

if ll[0] == 1:
    print("ll[0] is 1")
else:
    print("ll[0] is not 1")

# Condition

print(ll[0] == 1)

print(ll[1] >= 1)

print(1 not in ll)

# numpy

a = np.array([1, 2, 3])

print(a)

print(a.shape)

b = np.array([[1, 2, 3], [4, 5, 6]])

# slice: a[start:stop]   items start through stop-1

print(b[0, 0:2])

# fancy index

print(b[:, 0:2])

print(b[1, :])
