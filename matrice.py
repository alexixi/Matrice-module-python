"""
Module pour utiliser les matrices en Python.
La classe Matrice représente une matrice.
Une Matrice doit être créée avec une liste de liste de nombres.

Exemple :
    matrice = Matrice([[5, 6, 1], [3, 6, 9], [1, 1, 1]])
"""

from fractions import Fraction
import math
import random

def simplify_number_type(number: int | float | Fraction) -> int | Fraction:
    """
    Converti un float en Fraction
    Converti une fraction en int dans le cas des Fraction(0, x) et Fraction(x, 1)
    """
    if isinstance(number, float):
        number = Fraction(str(number))
    if isinstance(number, Fraction):
        if number.numerator == 0:
            return 0
        if number.denominator == 1:
            return number.numerator
    return number


class Matrice:
    def __init__(self, matrice: list[list[int | float | Fraction]] | list[list[int | float]] | list[list[int | Fraction]] | list[list[float | Fraction]] | list[list[int]] | list[list[float]] | list[list[Fraction]]) -> None:
        if not isinstance(matrice, list) or not all((isinstance(l, list) and len(l) == len(matrice[0])) for l in matrice):
            raise ValueError("La matrice n'est pas valide.\nType accepté pour la création : list[list[nombre]] avec nombre = int, float, Fraction\nLes sous listes doivent avoir le même nombre déléments")
        self.__tableau = [[simplify_number_type(e) for e in l] for l in matrice]

    @property
    def size(self) -> tuple[int, int]:
        return (len(self.__tableau), len(self.__tableau[0]))

    @property
    def n(self) -> int:
        return self.size[0]

    @property
    def m(self) -> int:
        return self.size[1]
    
    @property
    def is_square(self) -> bool:
        """Renvoi True si la matrice est carrée (i.e. n == m), False sinon"""
        return self.n == self.m

    def __getitem__(self, indice: int) -> int | Fraction:
        return self.__tableau[indice]
    
    def __eq__(self, attr) -> bool:
        """Deux matrices sont égales si leur tableau de valeurs sont égaux"""
        return self.__tableau == attr.__tableau if isinstance(attr, Matrice) else False

    def __len__(self) -> int:
        """Renvoi n"""
        return self.n

    def __iter__(self):
        """Renvoi l'objet iter() du tableau de valeurs"""
        return iter(self.__tableau)

    def __reversed__(self):
        """Renvoi l'objet reversed() du tableau de valeurs"""
        return reversed(self.__tableau)

    def __repr__(self) -> str:
        """Renvoi une chaîne de caractère pour représenter la matrice"""
        return f'Matrice({repr(self.__tableau)})'

    def __str__(self) -> str:
        """Renvoi une chaîne de caractère lisible pour un humain pour représenter la matrice"""
        return '\n'.join('  '.join(str(e) for e in l) for l in self.__tableau)
    
    def __nonzero__(self) -> bool:
        """Renvoi True si la matrice est non nulle, False sinon"""
        return not self.is_nulle()

    def __pos__(self) -> 'Matrice':
        return self.copy()

    def __neg__(self) -> 'Matrice':
        """Renvoie l'opposé de la matrice, où chaque élément est remplacé par son opposé"""
        return Matrice([[-e for e in l] for l in self.__tableau])

    def __abs__(self) -> 'Matrice':
        """Renvoie la matrice où chaque élément est remplacé par sa valeure absolue"""
        return Matrice([[abs(e) for e in l] for l in self.__tableau])

    def __add__(self, matrice: 'Matrice') -> 'Matrice':
        """Addition de deux matrices"""
        if not isinstance(matrice, Matrice):
            raise ValueError("Vous ne pouvez ajouter que deux matrices de même taille ensemble")
        if self.size != matrice.size:
            raise ValueError("Vous ne pouvez ajouter que deux matrices de même taille ensemble")
        return Matrice([[simplify_number_type(self[i][j] + matrice[i][j]) for j in range(self.m)] for i in range(self.n)])
    
    def __radd__(self, matrice: 'Matrice') -> 'Matrice':
        """Addition de deux matrices"""
        return self.__add__(matrice)

    def __sub__(self, matrice: 'Matrice') -> 'Matrice':
        """Soustraction de deux matrices"""
        if not isinstance(matrice, Matrice):
            raise ValueError("Vous ne pouvez soustraire que deux matrices de même taille ensemble")
        if self.size != matrice.size:
            raise ValueError("Vous ne pouvez soustraire que deux matrices de même taille ensemble")
        return Matrice([[simplify_number_type(self[i][j] - matrice[i][j]) for j in range(self.m)] for i in range(self.n)])
    
    def __rsub__(self, matrice: 'Matrice') -> 'Matrice':
        """Soustraction de deux matrices"""
        return self.__sub__(matrice)

    def __mul__(self, factor: 'Matrice' | int | float | Fraction) -> 'Matrice':
        """Multiplication de la matrice par un scalair ou une autre matrice"""
        if isinstance(factor, Matrice):
            n, p = self.size
            p2, m = factor.size
            if p2 != p:
                raise ValueError("Les matrices ne peuvent pas être multipliés")
            return Matrice([[simplify_number_type(sum(self[i][k] * factor[k][j] for k in range(p))) for j in range(m)] for i in range(n)])
        return Matrice([[simplify_number_type(e * factor) for e in l] for l in self])

    def __rmul__(self, factor: 'Matrice' | int | float | Fraction) -> 'Matrice':
        """Multiplication de la matrice par un scalair ou une autre matrice"""
        if isinstance(factor, Matrice):
            return factor.__mul__(self)
        return self.__mul__(factor)

    def __pow__(self, factor: int | float | Fraction) -> 'Matrice':
        """
        Calcul la matrice à la puissance factor. factor doit être un entier (il peut néanmoins être de type float (ex: 3.0) ou de type Fraction (ex: Fraction(3, 1))).
        Si la matrice est diagonale calcul directement la puissance factor, sinon calcul par recurrence chaque multiplication de la matrice avec elle même
        """
        if not self.is_square:
            raise ValueError('La matrice doit être carrée pour être élevée à une puissance')
        factor = simplify_number_type(factor)
        if not isinstance(factor, int):
            raise TypeError("La puissance doit être un nombre entier")
        if factor == 0:
            return create_id_matrice(self.n)
        if factor == 1:
            return self.copy()
        if factor == 0:
            return create_id_matrice(self.n)
        if self.is_diagonale():
            return Matrice([[e**factor for e in l] for l in self.__tableau])
        result = self.copy()
        for _ in range(factor-1):
            result = result * self.copy()
        return result
    
    def __truediv__(self, denominator) -> 'Matrice':
        """
        Équivalent à la multiplication par l'inverse
        Exemples: 
            matriceA/matriceB est équivalent à matriceA * matriceB.inverse()
            matrice/3 est équivalent à matrice * Fraction(1, 3)
        """
        if isinstance(denominator, Matrice):
            return self.__mul__(denominator.inverse())
        return Matrice([[simplify_number_type(Fraction(e, denominator)) for e in l] for l in self])
    
    def __rtruediv__(self, numerator) -> 'Matrice':
        """
        Équivalent à la multiplication par l'inverse
        Exemples: 
            matriceA/matriceB est équivalent à matriceA * matriceB.inverse()
            3/matrice est équivalent à 3 * matrice.inverse()
        """
        if isinstance(numerator, Matrice):
            return numerator.__mul__(self.inverse())
        return self.inverse().__mul__(numerator)
    
    def copy(self) -> 'Matrice':
        """Renvoi une copie de la matrice"""
        return Matrice(self.__tableau)

    def is_nulle(self) -> bool:
        """Renvoi True si la matrice est nulle, c'est à dire qu'elle est constituée uniquement de 0"""
        return all(e == 0 for l in self.__tableau for e in l)

    def is_diagonale(self) -> bool:
        """Renvoi True si la matrice est diagonale, c'est à dire une matrice avec des éléments uniquement sur la diagonale principale et des 0 sinon"""
        for i in range(self.n):
            for j in range(self.m):
                if i != j and self.__tableau[i][j] != 0:
                    return False
        return True

    def is_triangulaire_inf(self) -> bool:
        """Renvoi True si la matrice est triangulaire inférieur, c'est à dire une matrice ou le dessus de la diagonale principale est uniquement constituée de 0"""
        for i in range(self.n):
            for j in range(self.m):
                if i < j and self.__tableau[i][j] != 0:
                    return False
        return True

    def is_triangulaire_sup(self) -> bool:
        """Renvoi True si la matrice est triangulaire supérieure, c'est à dire une matrice ou le dessous de la diagonale principale est uniquement constituée de 0"""
        for i in range(self.n):
            for j in range(self.m):
                if i > j and self.__tableau[i][j] != 0:
                    return False
        return True

    def trace(self) -> int | Fraction:
        """Calcule la trace de la matrice, soit la somme des éléments de la diagonale principale"""
        return simplify_number_type(sum(self.__tableau[i][i] for i in range(self.n)))

    def transpose(self) -> 'Matrice':
        """Renvoi la transposée de la matrice, i.e. met les lignes en colonnes"""
        t = create_zero_matrice(self.m, self.n)
        for i in range(self.n):
            for j in range(self.m):
                t[j][i] = self.__tableau[i][j]
        return t

    def matrice_extraite(self, i: int, j: int) -> 'Matrice':
        """
        Renvoie la matrice de taille (n-1, m-1) obtenue en supprimant la ligne i et la colonne j (en considérant (n, m) la taille de la matrice d'origine)
        Fonction sans effet de bord (ne modifie pas la matrice d'origine)

        Args:
            matrice (MatriceType): Liste de liste d'entiers représentant la matrice
            i (int): Ligne à supprimer (en langage naturel, correspond à l'indice i-1 de la liste)
            j (int): Colonne à supprimer (en langage naturel, correspond à l'indice j-1 de la liste)

        Returns:
            Matrice: La matrice de taille (n-1, m-1) obtenue en supprimant la ligne i et la colonne j
        
        Raises:
            ValueError: Les numéros de ligne ou de colonne ne sont pas valides
        """
        if type(i) != int:
            raise ValueError("Le numéro de ligne à supprimer n'est pas un entier")
        if type(j) != int:
            raise ValueError("Le numéro de colonne à supprimer n'est pas un entier")
        if self.n == 1 or self.m == 1:
            raise ValueError("La matrice est trop petite")
        if i < 1 or i > self.n:
            raise ValueError("Le numéro de ligne à supprimer n'est pas dans la matrice")
        if j < 1 or j > self.m:
            raise ValueError("Le numéro de colonne à supprimer n'est pas dans la matrice")
        return Matrice([self.__tableau[indice_ligne][:j-1] + self.__tableau[indice_ligne][j:] for indice_ligne in range(self.m) if indice_ligne != i-1])
    
    def det(self) -> int | Fraction:
        """
        Calcule le déterminant d'une matrice carré de taille nxn quelconque en développant selon la colonne 1
        Fonction récursive
        La règle de Sarrus est utilisée pour calculer les déterminant 3x3 (cas de base pour la récursivité) et ainsi gagner du temps de calcul pour les grandes matrices
        Pour les cas spécifiques des déterminants de matrice 2x2 et 1x1 des méthodes directes sont utilisées

        Returns:
            int: Le déterminant de la matrice
        
        Raises:
            ValueError: La matrice n'est pas carré
        """
        if not self.is_square:
            raise ValueError("La matrice n'est pas carré")
        if self.n == 1:
            return self.__tableau[0][0]
        if self.n == 2:
            return simplify_number_type(self.__tableau[0][0] * self.__tableau[1][1] - self.__tableau[1][0] * self.__tableau[0][1])
        if self.n == 3:
            # regle de Sarrus
            return simplify_number_type(
                + self.__tableau[0][0] * self.__tableau[1][1] * self.__tableau[2][2]
                + self.__tableau[0][1] * self.__tableau[1][2] * self.__tableau[2][0]
                + self.__tableau[1][0] * self.__tableau[2][1] * self.__tableau[0][2]
                - self.__tableau[2][0] * self.__tableau[1][1] * self.__tableau[0][2]
                - self.__tableau[2][1] * self.__tableau[1][2] * self.__tableau[0][0]
                - self.__tableau[1][0] * self.__tableau[0][1] * self.__tableau[2][2]
            )
        if self.is_nulle():
            return 0
        if self.is_triangulaire_inf() or self.is_triangulaire_sup():
            product = 1
            for i in range(len(self)):
                product *= self.__tableau[i][i]
            return simplify_number_type(product)
        j = 1
        return sum(self.__tableau[i-1][j-1] * self.matrice_extraite(i, j).det() * (-1)**(i+j) for i in range(1, self.n+1))
    
    def commatrice(self) -> 'Matrice':
        """Renvoi la commatrice de la matrice"""
        com = create_zero_matrice(self.n, self.m)
        for i in range(1, self.n+1):
            for j in range(1, self.m+1):
                com[i-1][j-1] = (-1)**(i+j) * self.matrice_extraite(i, j).det()
        return com

    def inverse(self) -> 'Matrice':
        """Inverse la matrice en utilisant la méthode de la commatrice. Pour être inversible la matrice doit être carré et avoir un déterminant non nul."""
        determinant = self.det()
        if determinant == 0:
            raise ValueError("Matrice non inversible !")
        return self.commatrice().transpose() * Fraction(1, determinant)


def create_zero_matrice(n: int, m: int) -> Matrice:
    """
    Renvoie une matrice de taille nxm remplie de 0

    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes

    Returns:
        Matrice: La matrice nulle
    """
    return Matrice([[0 for _ in range(m)] for _ in range(n)])

def create_id_matrice(n: int) -> Matrice:
    """
    Renvoie une matrice identité de taille nxn, avec des 1 sur la diagonale principale et des 0 ailleurs

    Args:
        n (int): Nombre de lignes et de colonne (la matrice est carrée)

    Returns:
        Matrice: La matrice identité
    """
    return Matrice([[1 if j == i else 0 for j in range(n)] for i in range(n)])


def create_random_matrice(n: int, m: int, nb_min: int = 0, nb_max: int = 9) -> Matrice:
    """
    Renvoie une matrice de taille nxm avec des nombres aléatoires compris entre n_min et n_max (inclus)

    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        nb_min (int): Borne inférieure de l'ensemble dans lequel les nombres sont tirés aléatoirement
        nb_max (int): Borne supérieure de l'ensemble dans lequel les nombres sont tirés aléatoirement
    Returns:
        Matrice: Une matrice nxm remplie de nombres aléatoire entre nb_min et nb_max
    """
    return Matrice([[random.randint(nb_min, nb_max) for _ in range(m)] for _ in range(n)])

if __name__ == "__main__":
    matrice = Matrice([[1, 2], [3, 4]])
    matrice2 = Matrice([[1.5, 6, 1], [3, 6, 9], [1, 1, 1]])
    diag = Matrice([[5, 0, 0], [0, Fraction(1, 6), 0], [0, 0, 22]])
    troll = create_random_matrice(3, 3)
    trolll = create_random_matrice(3, 3)
    idd = create_id_matrice(3)
    print(troll*idd == troll)
    print(matrice2.det())
    print(1/matrice2)
    print(matrice**-1)