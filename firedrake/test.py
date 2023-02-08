import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

a = input()
b = input()

# Write an answer using print
# To debug: print("Debug messages...", file=sys.stderr, flush=True)


difference = ord(b[0]) - ord(a[0])
if difference > 26:
    difference -= 26
print(difference)
truth = 1
for i in range(len(a)):
    print(ord(b[i]) - ord(a[i]))
    if ((ord(b[i]) - ord(a[i])) % 26) == difference % 26:
        truth = 0
if truth == 1:
    print("true")
else:
    print('false')