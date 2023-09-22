import matrice

print("Inversez la matrice suivante :")

mat = matrice.create_random_matrice(3, 3, 0)
print(mat)

input("Pour donner la solution tappez n'importe quoi ")

print(mat.inverse_comatrice())
