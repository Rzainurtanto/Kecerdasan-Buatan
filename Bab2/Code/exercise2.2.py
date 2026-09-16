def sortarray(xs):
    n = len(xs)
    for i in range(n):
        minidx = i
        for j in range(i + 1, n):
            if xs[j] < xs[minidx]:
                minidx = j
        xs[i], xs[minidx] = xs[minidx], xs[i]
    return xs

data = [12, 5, 8, 130, 44, 3, 77]
print("Array setelah diurutkan:", sortarray(data))