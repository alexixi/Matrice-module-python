import matrice

mat = matrice.create_random_matrice(3, 3, 0)
print(mat)

det = mat.det()

reponse = input("Déterminant de la matrice : ")

if str(det) == reponse:
    print("Bonne réponse !")
else:
    print("Mauvais réponse !")
print("Le déterminant était", det)