def minarray(xs):
    minval = xs[0]
    for x in xs:
        if x < minval:
            minval = x
    return minval

data = [12, 5, 8, 130, 44, 3, 77]
print("Nilai minimum dari array adalah:", minarray(data))