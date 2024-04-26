import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
from collections import Counter
import copy
import collections


forget = [3, 4, 6, 7, 10, 13, 14, 15, 16, 17, 18, 23, 25, 37, 38, 39,
          40, 41, 42, 43, 44, 48, 53, 54, 58]
torch.save(forget, "nokeep.pth")

TOL_MISSING_DICT = [
    (9, 6), (10, 6), (12, 5), (13, 3), (13, 7), (14, 9), (21, 6),
    (21, 8), (22, 4), (33, 2), (39, 5), (40, 2), (41, 1), (43, 4),
    (43, 5), (44, 9), (24, 2), (56, 2)
]


def loc(tuple):
    a, b = tuple
    return (a-1) * 9 + b - 1


def get_index(loc):
    return (int((loc // 9) + 1), int((loc % 9) + 1))


plusidee = [[[get_index(j)] for j in range((i-1)*9 - 1, (i) * 9)] for i in forget]


to_delete = torch.load("delete.pth")
arr = torch.load("local.pth")
bads = torch.load("badsAll.pth")
flattened_list = [item for sublist in arr for item in sublist]
counter = Counter(flattened_list)
fruits, counts = zip(*counter.items())
sorted_fruits, sorted_counts = zip(*sorted(zip(fruits, counts), reverse=True))
sorted_fruits = np.array(sorted_fruits)

sorted_counts = np.array(sorted_counts)

for i in sorted_fruits:
    assert i in flattened_list

tol = 60
torch.save(sorted_fruits[:tol], 'toDrop.pth')
for j, sublist in enumerate(arr):
    sublist = list(set(sublist))
    index = []
    for i, string in enumerate(sublist):
        if string in sorted_fruits[:tol]:
            index.append(i)
    for i in list(reversed(index)):
        del arr[j][i]
length = [len(li) for li in arr]
for tuple in TOL_MISSING_DICT:
    length[loc(tuple)] = -1


indices = []
for i in np.nonzero(length)[0]:
    indices += [get_index(i)]
indices += TOL_MISSING_DICT
plusidee = list(itertools.chain.from_iterable(itertools.chain.from_iterable(plusidee)))
indices += plusidee
print(indices)
indices = sorted(list(set(indices)))
indices = np.array(indices)
firstval = indices[:, 0]
counter = collections.Counter(firstval)
print(indices)
for value, count in counter.items():
    print(f'{value}: {count}')

torch.save(indices, "nogo.pth")

bins = np.arange(-1, 6)
counts, _ = np.histogram(length, bins=bins)
print(counts)
print(sum(counts))
plt.bar(bins[:-1], counts)
plt.show()
plt.bar(range(len(length)), length)
plt.xlabel('Fruits')
plt.ylabel('Frequency')
plt.title('Fruit Frequency Histogram')
plt.show()

avoid = np.concatenate(
    (np.array([9, 10, 12, 13, 14, 21, 22, 24, 33, 39, 40, 41, 43, 44, 56]),
     to_delete),
    axis=0
)
lengths = np.array([len(sublist) for sublist in arr])
newavoid = np.concatenate(
    (np.where(lengths > 10)[0], avoid),
    axis=0
)
