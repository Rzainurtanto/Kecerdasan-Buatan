import math

n = int(input("Masukkan banyak angka yang akan dimasukkan: "))
angka = []

for i in range(n):
    nilai = float(input("Masukkan angka ke-" + str(i + 1) + ": "))
    angka.append(nilai)

for x in angka:
    hasil = math.sin(x)
    print("sin(", x, ") =", hasil)