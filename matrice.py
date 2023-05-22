"""
Module utilitaire pour faire des opérations sur les matrices et les déterminants
Les matrices sont représentées par le type MatriceType, une liste de liste d'entiers de réels ou de fractions rationnelles
"""

import random
import math
from fractions import Fraction

test_matrice_5x5 = [[32,4,6,1,3],[5,12,32,1,6],[7,8,9,6,5],[4,6,4,2,5],[3,5,6,12,63]]

test_matrice_4x4 = [
    [1,3,5,5],
    [2,65,8,6],
    [54,4,6,7],
    [5,6,5,3]
]

test_matrice_2x3 = [
    [1, 2, 3],
    [4, 5, 6]
]

test_matrice_3x3 = [
    [1, 1, 3],
    [4, 5, 6],
    [7, 8, 9]
]

test_matrice_triang_sup = [
    [1, 1, 3],
    [0, 5, 6],
    [0, 0, 9]
]

test_matrice_triang_inf = [
    [1, 0, 0],
    [4, 5, 0],
    [7, 8, 9]
]
test_matrice_diag = [
    [1, 0, 0],
    [0, 5, 0],
    [0, 0, 9]
]

test_matrice_2x2 = [
    [1, 1],
    [4, 8]
]

MatriceType = list[list[int | float | Fraction]] | list[list[int | float]] | list[list[int | Fraction]] | list[list[float | Fraction]] | list[list[int]] | list[list[float]] | list[list[Fraction]]

def affiche_matrice(matrice: MatriceType) -> None:
    print('\n'.join('  '.join(str(e) for e in l) for l in matrice))


def create_zero_matrice(n: int, m: int) -> list[list[int]]:
    """
    Renvoie une matrice de taille nxm remplie de 0

    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes

    Returns:
        list[list[int]]: La matrice vide
    """
    return [[0 for _ in range(m)] for _ in range(n)]


def create_random_matrice(n: int, m: int, nb_min: int = 0, nb_max: int = 10) -> list[list[int]]:
    """
    Renvoie une matrice de taille nxm avec des nombres aléatoires compris entre n_min et n_max (inclus)

    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        nb_min (int): Borne inférieure de l'ensemble dans lequel les nombres sont tirés aléatoirement
        nb_max (int): Borne supérieure de l'ensemble dans lequel les nombres sont tirés aléatoirement
    Returns:
        list[list[int]]: Une matrice nxm remplie de nombres aléatoire entre n_min et n_max
    """
    return [[random.randint(nb_min, nb_max) for _ in range(m)] for _ in range(n)]


def is_matrice_valide(matrice: MatriceType) -> bool:
    """
    Vérifie que les lignes ont toute la même taille. Si ce n'est pas le cas la matrice n'est pas valide

    Args:
        matrice (MatriceType): Liste de liste d'entiers représentant la matrice
    
    Returns:
        bool: True si la matrice est valide, False sinon
    
    Exemples:
        >>> is_matrice_valide([[1, 2], [3, 4]])
        True

        >>> is_matrice_valide([[4, 2], [4, 4], [12, 6]])
        True

        >>> is_matrice_valide([[1, 2, 3], [4, 5]])
        False

        >>> is_matrice_valide([[4, 23], [11], [5, 6]])
        False
    """
    # if type(matrice) != MatriceType:
    #     return False
    n = len(matrice[0])
    return all(len(l) == n for l in matrice)


def taille_matrice(matrice: MatriceType) -> tuple[int, int]:
    """
    Donne la taille de la matrice sous forme d'un tuple (n, m) avec n le nombre de lignes et m le nombre de colonnes

    Args:
        matrice (MatriceType): Liste de liste d'entiers représentant la matrice
    
    Returns:
        tuple[int, int]: (n, m) avec n le nombre de lignes et m le nombre de colonnes

    Raises:
        ValueError: la matrice n'est pas valide

    Exemples:
        >>> taille_matrice([[1, 2], [3, 4]])
        (2, 2)

        >>> taille_matrice([[1, 2], [3, 4], [5, 6]])
        (3, 2)

        >>> taille_matrice([[12, 3, 7], [21, 8, 2]])
        (2, 3)

    """
    if not is_matrice_valide(matrice):
        raise ValueError("La matrice n'est pas valide")
    return (len(matrice), len(matrice[0]))

def scalaire_product(matrice: MatriceType, number: float) -> MatriceType:
    return [[e * number for e in l] for l in matrice]

def sum_matrice(matrice_1: MatriceType, matrice_2: MatriceType) -> MatriceType:
    n, m = taille_matrice(matrice_1)
    if (n, m) != taille_matrice(matrice_2):
        raise ValueError("Les deux matrices n'ont pas la même taille")
    return [[matrice_1[i][j] + matrice_2[i][j] for j in range(m)] for i in range(n)]

def matrice_product(matrice_1: MatriceType, matrice_2: MatriceType) -> MatriceType:
    n, p = taille_matrice(matrice_1)
    p2, m = taille_matrice(matrice_1)
    if p2 != p:
        raise ValueError("Les matrices ne peuvent pas être multipliés")
    return [[sum(matrice_1[i][k] * matrice_2[k][j] for k in range(p)) for j in range(m)] for i in range(n)]

affiche_matrice(matrice_product([[Fraction(1, 2), 2], [3, 4]], [[2, 3], [5, 7]]))

def transpose(matrice: MatriceType) -> MatriceType:
    n, m = taille_matrice(matrice)
    t = create_zero_matrice(m, n)
    for i in range(n):
        for j in range(m):
            t[j][i] = matrice[i][j]
    return t

def trace(matrice: MatriceType) -> float:
    """
    Calcule la trace de la matrice, soit la somme des éléments de la diagonale principale
    """
    return sum(matrice[i][i] for i in range(len(matrice)))


def det_2x2(matrice: MatriceType) -> float:
    """
    Calcule le déterminant pour une matrice de taille 2x2

    Args:
        matrice (MatriceType): Liste de liste d'entiers représentant la matrice
    
    Returns:
        int: Le déterminant de la matrice

    Raises:
        ValueError: la matrice n'est pas de taille 2x2
    
    Exemples:
        >>> det_2x2([[1, 2], [3, 4]])
        -2

        >>> det_2x2([[0, 0], [13, 11]])
        0

        >>> det_2x2([[42, 49], [41, 14]])
        -1421

    """
    n, m = taille_matrice(matrice)
    if n != 2 or m != 2:
        raise ValueError("La matrice n'est pas de taille 2x2")
    return matrice[0][0] * matrice[1][1] - matrice[1][0] * matrice[0][1]


def det_3x3_sarrus(matrice: MatriceType) -> float:
    """
    Calcule le déterminant pour une matrice de taille 3x3 en utilisant la règle de Sarrus

    Args:
        matrice (MatriceType): Liste de liste d'entiers représentant la matrice
    
    Returns:
        int: Le déterminant de la matrice
    
    Exemples:
        >>> det_3x3_sarrus([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        0

        >>> det_3x3_sarrus([[1, 1, 3], [4, 5, 6], [7, 8, 9]])
        -6

        >>> det_3x3_sarrus([[6, 3, 33], [48, 21, 12], [20, 44, 17]])
        53082
    """
    n, m = taille_matrice(matrice)
    if n != 3 or m != 3:
        raise ValueError("La matrice n'est pas de taille 3x3")
    return (
        matrice[0][0] * matrice[1][1] * matrice[2][2]
        + matrice[0][1] * matrice[1][2] * matrice[2][0]
        + matrice[1][0] * matrice[2][1] * matrice[0][2]
        - matrice[2][0] * matrice[1][1] * matrice[0][2]
        - matrice[2][1] * matrice[1][2] * matrice[0][0]
        - matrice[1][0] * matrice[0][1] * matrice[2][2]
    )


def is_matrice_diagonale(matrice: MatriceType) -> bool:
    """
    Détermine si la matrice est diagonale

    Args:
        matrice (MatriceType): Liste de liste d'entiers représentant la matrice
    
    Returns:
        bool: True si la matrice est diagonale, False sinon
    
    Exemples:
        >>> is_matrice_diagonale([[2, 0, 0], [0, 5, 0], [0, 0, 9]])
        True

        >>> is_matrice_diagonale([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        False

        >>> is_matrice_diagonale([[1, 0], [1, 1]])
        False

        >>> is_matrice_diagonale([[1, 0, 0], [4, 5, 0], [7, 8, 9]])
        False
    """
    n, m = taille_matrice(matrice)
    for i in range(n):
        for j in range(m):
            if i != j and matrice[i][j] != 0:
                return False
    return True

def is_matrice_triangulaire_sup(matrice: MatriceType) -> bool:
    """
    Détermine si la matrice est triangulaire supérieure

    Args:
        matrice (MatriceType): Liste de liste d'entiers représentant la matrice
    
    Returns:
        bool: True si la matrice est triangulaire supérieure, False sinon
    
    Exemples:
        >>> is_matrice_triangulaire_sup([[1, 4, 3], [0, 5, 12], [0, 0, 9]])
        True

        >>> is_matrice_triangulaire_sup([[2, 0, 0], [0, 5, 0], [0, 0, 9]])
        True

        >>> is_matrice_triangulaire_sup([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        False

        >>> is_matrice_triangulaire_sup([[1, 1], [0, 1]])
        True

        >>> is_matrice_triangulaire_sup([[1, 0, 0], [4, 5, 0], [1, 18, 4]])
        False
    """
    n, m = taille_matrice(matrice)
    for i in range(n):
        for j in range(m):
            if i > j and matrice[i][j] != 0:
                return False
    return True

def is_matrice_triangulaire_inf(matrice: MatriceType) -> bool:
    """
    Détermine si la matrice est triangulaire inférieure

    Args:
        matrice (MatriceType): Liste de liste d'entiers représentant la matrice
    
    Returns:
        bool: True si la matrice est triangulaire inférieure, False sinon
    
    Exemples:
        >>> is_matrice_triangulaire_inf([[1, 0, 0], [4, 5, 0], [1, 18, 4]])
        True

        >>> is_matrice_triangulaire_inf([[2, 0, 0], [0, 5, 0], [0, 0, 9]])
        True

        >>> is_matrice_triangulaire_inf([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        False

        >>> is_matrice_triangulaire_inf([[1, 0], [1, 1]])
        True

        >>> is_matrice_triangulaire_inf([[1, 4, 3], [0, 5, 12], [0, 0, 9]])
        False
    """
    n, m = taille_matrice(matrice)
    for i in range(n):
        for j in range(m):
            if i < j and matrice[i][j] != 0:
                return False
    return True

def det_triangulaire(matrice: MatriceType) -> float:
    """
    Calcule le déterminant pour une matrice diagonale, triangulaire inférieur ou triangulaire supérieure, c'est à dire le produit des éléments de la diagonale principale.
    Ne fonctionne que pour une matrice triangulaire !

    Args:
        matrice (MatriceType): Liste de liste d'entiers représentant la matrice

    Returns:
        int: Le déterminant de la matrice triangulaire

    Exemples:
        >>> det_triangulaire([[2, 0, 0], [0, 5, 0], [0, 0, 9]])
        90
        >>> det_triangulaire([[1, 4, 3], [0, 5, 12], [0, 0, 9]])
        45
        >>> det_triangulaire([[1, 0, 0], [4, 5, 0], [1, 18, 4]])
        20
    """
    product = 1
    for i in range(len(matrice)):
        product *= matrice[i][i]
    return product

def suppr_ligne_colonne(matrice: MatriceType, i: int, j: int) -> MatriceType:
    """
    Renvoie la matrice de taille (n-1, m-1) obtenue en supprimant la ligne i et la colonne j (en considérant (n, m) la taille de la matrice d'origine)
    Fonction sans effet de bord (ne modifie pas la matrice d'origine)

    Args:
        matrice (MatriceType): Liste de liste d'entiers représentant la matrice
        i (int): Ligne à supprimer (en langage naturel, correspond à l'indice i-1 de la liste)
        j (int): Colonne à supprimer (en langage naturel, correspond à l'indice j-1 de la liste)

    Returns:
        MatriceType: La matrice de taille (n-1, m-1) obtenue en supprimant la ligne i et la colonne j
    
    Raises:
        ValueError: Les numéros de ligne ou de colonne ne sont pas valides

    Exemples:
        >>> suppr_ligne_colonne([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1, 1)
        [[5, 6], [8, 9]]

        >>> suppr_ligne_colonne([[3, 2, 4, 3], [2, 10, 0, 8], [7, 9, 5, 10], [10, 4, 4, 0]], 3, 3)
        [[3, 2, 3], [2, 10, 8], [10, 4, 0]]

        >>> suppr_ligne_colonne([[1, 4, 3], [0, 5, 12], [0, 0, 9]], 3, 3)
        [[1, 4], [0, 5]]

        >>> suppr_ligne_colonne([[1, 4], [6, 2]], 1, 1)
        [[2]]
    """
    if type(i) != int:
        raise ValueError("Le numéro de ligne à supprimer n'est pas un entier")
    if type(j) != int:
        raise ValueError("Le numéro de colonne à supprimer n'est pas un entier")
    n, m = taille_matrice(matrice)
    if n == 1 or m == 1:
        raise ValueError("La matrice est trop petite")
    if i < 1 or i > n:
        raise ValueError("Le numéro de ligne à supprimer n'est pas dans la matrice")
    if j < 1 or j > m:
        raise ValueError("Le numéro de colonne à supprimer n'est pas dans la matrice")
    return [matrice[indice_ligne][:j-1] + matrice[indice_ligne][j:] for indice_ligne in range(m) if indice_ligne != i-1]


def det(matrice: MatriceType) -> float:
    """
    Calcule le déterminant d'une matrice carré de taille nxn quelconque en développant selon la colonne 1
    Fonction récursive
    La règle de Sarrus est utilisée pour calculer les déterminant 3x3 (cas de base pour la récursivité) et ainsi gagner du temps de calcul pour les grandes matrices
    Pour les cas spécifiques des déterminants de matrice 2x2 et 1x1 des méthodes directes sont utilisées

    Args:
        matrice (MatriceType): Liste de liste d'entiers représentant la matrice

    Returns:
        int: Le déterminant de la matrice
    
    Raises:
        ValueError: La matrice n'est pas carré
    
    Exemples:
        >>> det([[32, 4, 6, 1, 3],[5, 12, 32, 1, 6],[7, 8, 9, 6, 5],[4, 6, 4, 2, 5],[3, 5, 6, 12, 63]])
        1010327

        >>> det([[3, 2, 4, 3], [2, 10, 0, 8], [7, 9, 5, 10], [10, 4, 4, 0]])
        636

        >>> det([[1, 1, 3], [4, 5, 6], [7, 8, 9]])
        -6

        >>> det([[1, 4, 3], [0, 5, 12], [0, 0, 9]])
        45

        >>> det([[1, 2], [3, 4]])
        -2

        >>> det([[11]])
        11

    """
    n, m = taille_matrice(matrice)
    if n != m:
        raise ValueError("La matrice n'est pas carré")
    if n == 1:
        return matrice[0][0]
    if n == 2:
        return det_2x2(matrice)
    if n == 3:
        return det_3x3_sarrus(matrice)
    if is_matrice_triangulaire_inf(matrice) or is_matrice_triangulaire_sup(matrice):
        return det_triangulaire(matrice)
    j = 1
    return sum(matrice[i-1][j-1] * det(suppr_ligne_colonne(matrice, i, j)) * (-1)**(i+j) for i in range(1, n+1))

def commatrice(matrice: MatriceType) -> MatriceType:
    n, m = taille_matrice(matrice)
    com = create_zero_matrice(n, m)
    for i in range(1, n+1):
        for j in range(1, m+1):
            com[i-1][j-1] = (-1)**(i+j) * det(suppr_ligne_colonne(matrice, i, j))
    return com

def inverse(matrice: MatriceType) -> MatriceType:
    n, m = taille_matrice(matrice)
    determinant = det(matrice)
    if determinant == 0:
        raise ValueError("Matrice non inversible !")
    return scalaire_product(transpose(commatrice(matrice)), Fraction(1, det(matrice)))

# print(inverse(test_matrice_2x2))
affiche_matrice(matrice_product(inverse(test_matrice_5x5), test_matrice_5x5))
    


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # import time
    # mat = [[1, 0], [0, 4]]
    # mat = create_random_matrice(10, 10)
    # start = time.time()
    # print(det(mat))
    # print(time.time() - start)