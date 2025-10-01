# 🐍 Python DSA Quick Reference Cheatsheet

> A comprehensive guide to Python data structures and algorithms for technical interviews

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📚 Table of Contents

1. [Lists (Dynamic Arrays)](#1-lists-dynamic-arrays)
2. [Strings](#2-strings)
3. [Tuples](#3-tuples-immutable)
4. [Sets](#4-sets)
5. [Dictionaries (Hash Maps)](#5-dictionaries-hash-maps)
6. [Deque (Double-ended Queue)](#6-deque-double-ended-queue)
7. [Heap (Priority Queue)](#7-heap-priority-queue)
8. [Counter](#8-counter)
9. [DefaultDict](#9-defaultdict)
10. [Stack Operations](#10-stack-operations)
11. [Queue Operations](#11-queue-operations)
12. [OrderedDict](#12-ordereddict)
13. [Sorting](#13-sorting)
14. [Binary Search](#14-binary-search)
15. [Itertools](#15-itertools-common-patterns)
16. [Math & Number Operations](#16-math--number-operations)
17. [Common Built-in Functions](#17-common-built-in-functions)
18. [Two Pointers Pattern](#18-two-pointers-pattern)
19. [Sliding Window Pattern](#19-sliding-window-pattern)
20. [Useful Code Snippets](#20-useful-code-snippets)
21. [Time Complexity Reference](#time-complexity-quick-reference)

---

## 1. Lists (Dynamic Arrays)

### Creation & Basic Operations

```python
arr = []                    # Empty list
arr = [1, 2, 3]            # Initialized list
arr = [0] * n              # List with n zeros
arr = list(range(5))       # [0, 1, 2, 3, 4]
```

### Common Methods

| Method | Description | Time Complexity |
|--------|-------------|-----------------|
| `arr.append(x)` | Add to end | O(1) |
| `arr.pop()` | Remove from end | O(1) |
| `arr.pop(i)` | Remove at index i | O(n) |
| `arr.insert(i, x)` | Insert x at index i | O(n) |
| `arr.remove(x)` | Remove first occurrence | O(n) |
| `arr.extend([4, 5])` | Add multiple elements | O(k) |
| `arr.reverse()` | Reverse in-place | O(n) |
| `arr.sort()` | Sort ascending | O(n log n) |
| `arr.sort(reverse=True)` | Sort descending | O(n log n) |
| `arr.count(x)` | Count occurrences | O(n) |
| `arr.index(x)` | Find first index | O(n) |
| `arr.clear()` | Remove all elements | O(n) |

### List Comprehension

```python
squares = [x**2 for x in range(10)]
evens = [x for x in arr if x % 2 == 0]
matrix = [[0]*m for _ in range(n)]  # n x m matrix
```

### Slicing

```python
arr[start:end]             # Elements from start to end-1
arr[start:]                # From start to end
arr[:end]                  # From beginning to end-1
arr[::step]                # Every step element
arr[::-1]                  # Reverse list
arr[:]                     # Shallow copy
```

---

## 2. Strings

### Creation & Basic Operations

```python
s = "hello"
s = str(123)               # "123"
s = ''.join(['a', 'b'])    # "ab"
```

### Common Methods

| Method | Description | Example |
|--------|-------------|---------|
| `s.upper()` | Convert to uppercase | `"HELLO"` |
| `s.lower()` | Convert to lowercase | `"hello"` |
| `s.strip()` | Remove whitespace | `"hello"` |
| `s.split()` | Split by whitespace | `['hello', 'world']` |
| `s.split(',')` | Split by delimiter | Custom split |
| `s.replace('l', 'L')` | Replace characters | `"heLLo"` |
| `s.find('ll')` | Find substring | Returns -1 if not found |
| `s.index('ll')` | Find substring | Raises error if not found |
| `s.startswith('he')` | Check prefix | `True` |
| `s.endswith('lo')` | Check suffix | `True` |
| `s.isalpha()` | Check if alphabetic | Boolean |
| `s.isdigit()` | Check if digit | Boolean |
| `s.isalnum()` | Check if alphanumeric | Boolean |
| `''.join(list)` | Join list into string | String |

### String Operations

```python
s[i]                       # Access character (strings are immutable)
s[start:end]               # Slicing
len(s)                     # Length
ord('a')                   # ASCII value (97)
chr(97)                    # Character from ASCII ('a')
```

---

## 3. Tuples (Immutable)

```python
t = (1, 2, 3)
t = 1, 2, 3                # Packing
a, b, c = t                # Unpacking
t.count(1)                 # Count occurrences
t.index(2)                 # Find index
```

**Use Cases**: Immutable data, dictionary keys, returning multiple values

---

## 4. Sets

### Creation & Basic Operations

```python
s = set()                  # Empty set
s = {1, 2, 3}              # Initialized set
s = set([1, 2, 2, 3])      # From list: {1, 2, 3}
```

### Common Methods

```python
s.add(x)                   # Add element - O(1)
s.remove(x)                # Remove (raises error if not found) - O(1)
s.discard(x)               # Remove (no error) - O(1)
s.pop()                    # Remove arbitrary element - O(1)
s.clear()                  # Remove all elements
x in s                     # Check membership - O(1)
len(s)                     # Size
```

### Set Operations

| Operation | Syntax | Alternative |
|-----------|--------|-------------|
| Union | `s1 \| s2` | `s1.union(s2)` |
| Intersection | `s1 & s2` | `s1.intersection(s2)` |
| Difference | `s1 - s2` | `s1.difference(s2)` |
| Symmetric Difference | `s1 ^ s2` | `s1.symmetric_difference(s2)` |
| Subset | - | `s1.issubset(s2)` |
| Superset | - | `s1.issuperset(s2)` |

---

## 5. Dictionaries (Hash Maps)

### Creation & Basic Operations

```python
d = {}                     # Empty dict
d = {'a': 1, 'b': 2}       # Initialized dict
d = dict(a=1, b=2)         # Using dict()
d = {x: x**2 for x in range(5)}  # Dict comprehension
```

### Common Methods

```python
d[key] = value             # Add/update - O(1)
d.get(key, default)        # Get with default - O(1)
d.pop(key, default)        # Remove and return - O(1)
d.popitem()                # Remove arbitrary pair - O(1)
del d[key]                 # Delete key - O(1)
key in d                   # Check existence - O(1)
d.keys()                   # All keys
d.values()                 # All values
d.items()                  # All (key, value) pairs
d.clear()                  # Remove all
d.update(d2)               # Merge dictionaries
d.setdefault(key, default) # Get or set default
```

### Iteration

```python
for key in d:              # Iterate keys
for value in d.values():   # Iterate values
for key, value in d.items():  # Iterate pairs
```

---

## 6. Deque (Double-ended Queue)

```python
from collections import deque

dq = deque()               # Empty deque
dq = deque([1, 2, 3])      # Initialized

dq.append(x)               # Add to right - O(1)
dq.appendleft(x)           # Add to left - O(1)
dq.pop()                   # Remove from right - O(1)
dq.popleft()               # Remove from left - O(1)
dq.extend([4, 5])          # Extend right
dq.extendleft([0, -1])     # Extend left
dq.rotate(n)               # Rotate n steps right
dq.reverse()               # Reverse in-place
dq[i]                      # Access by index - O(n)
len(dq)                    # Size
dq.clear()                 # Remove all
```

**Use Cases**: Queue, stack, sliding window problems

---

## 7. Heap (Priority Queue)

```python
import heapq

heap = []                  # Min heap by default
heapq.heappush(heap, x)    # Add element - O(log n)
heapq.heappop(heap)        # Remove min - O(log n)
heap[0]                    # Peek min - O(1)
heapq.heapify(list)        # Convert list to heap - O(n)
heapq.heappushpop(heap, x) # Push then pop - O(log n)
heapq.heapreplace(heap, x) # Pop then push - O(log n)

# N largest/smallest
heapq.nlargest(n, iterable)
heapq.nsmallest(n, iterable)
heapq.nlargest(n, iterable, key=lambda x: x[1])

# Max heap trick
heapq.heappush(heap, -x)   # Negate values
max_val = -heapq.heappop(heap)
```

**Use Cases**: Top K elements, median finding, Dijkstra's algorithm

---

## 8. Counter

```python
from collections import Counter

cnt = Counter()            # Empty counter
cnt = Counter([1, 1, 2, 3, 3, 3])  # From list
cnt = Counter("aabbbc")    # From string

cnt[x]                     # Count of x (returns 0 if not exists)
cnt.most_common(n)         # n most common elements
cnt.elements()             # Iterator over elements
cnt.update([4, 4])         # Add counts
cnt.subtract([1, 2])       # Subtract counts
cnt1 + cnt2                # Add counters
cnt1 - cnt2                # Subtract (keeps only positive)
cnt1 & cnt2                # Intersection (min)
cnt1 | cnt2                # Union (max)
```

**Use Cases**: Frequency counting, anagram problems

---

## 9. DefaultDict

```python
from collections import defaultdict

dd = defaultdict(int)      # Default value 0
dd = defaultdict(list)     # Default value []
dd = defaultdict(set)      # Default value set()
dd = defaultdict(lambda: [])  # Custom default

dd[key]                    # Auto-creates if not exists
```

**Use Cases**: Grouping, graph adjacency lists

---

## 10. Stack Operations

```python
# Use list as stack
stack = []
stack.append(x)            # Push - O(1)
stack.pop()                # Pop - O(1)
stack[-1]                  # Peek - O(1)
len(stack)                 # Size
not stack                  # Check if empty
```

**Use Cases**: DFS, parentheses matching, expression evaluation

---

## 11. Queue Operations

```python
from collections import deque

queue = deque()
queue.append(x)            # Enqueue - O(1)
queue.popleft()            # Dequeue - O(1)
queue[0]                   # Front - O(1)
len(queue)                 # Size
not queue                  # Check if empty
```

**Use Cases**: BFS, level-order traversal

---

## 12. OrderedDict

```python
from collections import OrderedDict

od = OrderedDict()         # Maintains insertion order
od[key] = value
od.move_to_end(key)        # Move to end
od.move_to_end(key, last=False)  # Move to beginning
od.popitem(last=True)      # LIFO order
od.popitem(last=False)     # FIFO order
```

**Note**: Regular dicts maintain insertion order in Python 3.7+

---

## 13. Sorting

```python
# Sort list
arr.sort()                 # In-place
sorted_arr = sorted(arr)   # Returns new list

# Sort with key
arr.sort(key=lambda x: x[1])
arr.sort(key=lambda x: (x[0], -x[1]))  # Multiple criteria

# Reverse sort
arr.sort(reverse=True)

# Sort strings
arr.sort(key=str.lower)    # Case-insensitive

# Custom comparator (using functools)
from functools import cmp_to_key
arr.sort(key=cmp_to_key(lambda x, y: x - y))
```

---

## 14. Binary Search

```python
import bisect

arr = [1, 3, 4, 4, 6, 8]

bisect.bisect_left(arr, x)    # Leftmost insertion point
bisect.bisect_right(arr, x)   # Rightmost insertion point
bisect.bisect(arr, x)         # Same as bisect_right
bisect.insort_left(arr, x)    # Insert maintaining order
bisect.insort_right(arr, x)   # Insert maintaining order
```

### Manual Binary Search

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

---

## 15. Itertools (Common Patterns)

```python
from itertools import *

# Combinations & Permutations
combinations([1,2,3], 2)      # [(1,2), (1,3), (2,3)]
permutations([1,2,3], 2)      # [(1,2), (1,3), (2,1), (2,3), (3,1), (3,2)]
product([1,2], [3,4])         # [(1,3), (1,4), (2,3), (2,4)]

# Infinite iterators
count(start, step)            # count(10, 2) -> 10, 12, 14, ...
cycle([1,2,3])                # 1, 2, 3, 1, 2, 3, ...
repeat(x, n)                  # x, x, x, ... (n times)

# Useful iterators
accumulate([1,2,3,4])         # [1, 3, 6, 10] (cumulative sum)
chain([1,2], [3,4])           # [1, 2, 3, 4]
groupby(iterable, key)        # Group consecutive elements
```

---

## 16. Math & Number Operations

```python
import math

# Basic operations
abs(x)                     # Absolute value
pow(x, y)                  # x^y
x ** y                     # Power
x // y                     # Floor division
x % y                      # Modulo
divmod(x, y)               # Returns (quotient, remainder)

# Math functions
math.floor(x)              # Floor
math.ceil(x)               # Ceiling
math.sqrt(x)               # Square root
math.gcd(a, b)             # Greatest common divisor
math.lcm(a, b)             # Least common multiple (Python 3.9+)
math.factorial(n)          # n!
math.log(x, base)          # Logarithm
math.log2(x)               # Log base 2

# Constants
math.inf                   # Infinity
-math.inf                  # Negative infinity
math.pi                    # Pi

# Min/max
min(a, b, c)
max(a, b, c)
min(arr)
max(arr)
```

---

## 17. Common Built-in Functions

```python
# Type conversion
int(x)
float(x)
str(x)
list(x)
tuple(x)
set(x)

# Aggregation
sum(arr)                   # Sum of elements
sum(arr, start)            # Sum with initial value
len(arr)                   # Length
min(arr)
max(arr)
all(arr)                   # True if all truthy
any(arr)                   # True if any truthy

# Functional
map(func, arr)             # Apply function
filter(func, arr)          # Filter elements
zip(arr1, arr2)            # Zip iterables
enumerate(arr)             # (index, value) pairs
reversed(arr)              # Reverse iterator
sorted(arr)                # Return sorted copy

# Range
range(n)                   # 0 to n-1
range(start, end)          # start to end-1
range(start, end, step)    # With step
```

---

## 18. Two Pointers Pattern

### Opposite Direction

```python
left, right = 0, len(arr) - 1
while left < right:
    if condition:
        left += 1
    else:
        right -= 1
```

### Same Direction (Fast & Slow)

```python
slow = fast = 0
while fast < len(arr):
    if condition:
        slow += 1
    fast += 1
```

**Use Cases**: Two sum, palindrome checking, removing duplicates

---

## 19. Sliding Window Pattern

### Fixed Size Window

```python
window_sum = sum(arr[:k])
max_sum = window_sum
for i in range(k, len(arr)):
    window_sum += arr[i] - arr[i-k]
    max_sum = max(max_sum, window_sum)
```

### Variable Size Window

```python
left = 0
for right in range(len(arr)):
    # Add arr[right] to window
    while condition_violated:
        # Remove arr[left] from window
        left += 1
    # Update result
```

**Use Cases**: Maximum subarray sum, longest substring problems

---

## 20. Useful Code Snippets

### Matrix Traversal

```python
# 4 directions (up, right, down, left)
directions = [(0,1), (1,0), (0,-1), (-1,0)]
for dx, dy in directions:
    nx, ny = x + dx, y + dy

# 8 directions (including diagonals)
directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
```

### Check Bounds

```python
def in_bounds(x, y, rows, cols):
    return 0 <= x < rows and 0 <= y < cols
```

### Binary Representation

```python
bin(x)                     # '0b1010'
bin(x)[2:]                 # '1010'
x.bit_count()              # Count set bits (Python 3.10+)
x & 1                      # Check if odd
x >> 1                     # Divide by 2
x << 1                     # Multiply by 2
x & (x-1)                  # Remove rightmost set bit
x | (1 << i)               # Set i-th bit
x & ~(1 << i)              # Clear i-th bit
x ^ (1 << i)               # Toggle i-th bit
```

### Input/Output Optimization

```python
# Fast input
import sys
input = sys.stdin.readline

# Multiple integers
a, b = map(int, input().split())

# List of integers
arr = list(map(int, input().split()))

# Multiple test cases
t = int(input())
for _ in range(t):
    # solve test case
    pass
```

### Graph Representation

```python
# Adjacency List
from collections import defaultdict
graph = defaultdict(list)
graph[u].append(v)

# Adjacency Matrix
n = 5
graph = [[0] * n for _ in range(n)]
graph[u][v] = 1
```

### DFS Template

```python
def dfs(node, visited):
    if node in visited:
        return
    visited.add(node)
    for neighbor in graph[node]:
        dfs(neighbor, visited)
```

### BFS Template

```python
from collections import deque

def bfs(start):
    queue = deque([start])
    visited = {start}
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

---

## Time Complexity Quick Reference

| Data Structure | Access | Search | Insert | Delete |
|----------------|--------|--------|--------|--------|
| **List** | O(1) | O(n) | O(n) | O(n) |
| **List (end)** | O(1) | - | O(1) | O(1) |
| **Set** | - | O(1)* | O(1)* | O(1)* |
| **Dict** | O(1)* | O(1)* | O(1)* | O(1)* |
| **Deque (both ends)** | O(1) | - | O(1) | O(1) |
| **Heap** | O(1) peek | O(n) | O(log n) | O(log n) |

\* Average case; worst case O(n) for hash collisions

### Common Algorithm Complexities

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| **Binary Search** | O(log n) | O(1) |
| **Merge Sort** | O(n log n) | O(n) |
| **Quick Sort** | O(n log n) avg | O(log n) |
| **Heap Sort** | O(n log n) | O(1) |
| **DFS/BFS** | O(V + E) | O(V) |
| **Dijkstra** | O((V + E) log V) | O(V) |

---

## 📖 Additional Resources

- [Python Official Documentation](https://docs.python.org/3/)
- [LeetCode](https://leetcode.com/)
- [HackerRank](https://www.hackerrank.com/)
- [GeeksforGeeks](https://www.geeksforgeeks.org/)

---

## 🤝 Contributing

Feel free to contribute by submitting pull requests or opening issues for improvements!

## 📄 License

This cheatsheet is available under the MIT License.

---

**⭐ Star this repository if you find it helpful!**

*Last Updated: October 2025*
