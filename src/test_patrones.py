"""
Test de Patrones para RPSAI
============================

Este script evalua el rendimiento del modelo contra diferentes
patrones de juego adversarios que podrian usarse para intentar
ganar al modelo o explotarlo.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from modelo import JugadorIA, JUGADA_A_NUM, NUM_A_JUGADA, GANA_A, PIERDE_CONTRA


# CONFIGURACION DEL TEST
RONDAS_POR_PARTIDA = 100
REPETICIONES_POR_PATRON = 50


class Oponente:
    """Clase base para diferentes estrategias de oponente"""

    def __init__(self):
        self.historial = []

    def registrar_ronda(self, jugada_ia, jugada_propia):
        self.historial.append((jugada_ia, jugada_propia))

    def decidir_jugada(self) -> str:
        raise NotImplementedError


class OponenteAleatorio(Oponente):
    """Juega completamente aleatorio"""

    def decidir_jugada(self) -> str:
        return np.random.choice(["piedra", "papel", "tijera"])


class OponentePiedraConstante(Oponente):
    """Siempre juega piedra"""

    def decidir_jugada(self) -> str:
        return "piedra"


class OponentePapelConstante(Oponente):
    """Siempre juega papel"""

    def decidir_jugada(self) -> str:
        return "papel"


class OponenteTijeraConstante(Oponente):
    """Siempre juega tijera"""

    def decidir_jugada(self) -> str:
        return "tijera"


class OponenteCiclo(Oponente):
    """Juega en ciclo: piedra -> papel -> tijera -> piedra..."""

    def decidir_jugada(self) -> str:
        ciclo = ["piedra", "papel", "tijera"]
        return ciclo[len(self.historial) % 3]


class OponenteCounterBot(Oponente):
    """Juega lo que le ganaria a la ultima jugada de la IA"""

    def decidir_jugada(self) -> str:
        if not self.historial:
            return np.random.choice(["piedra", "papel", "tijera"])

        ultima_ia = self.historial[-1][0]
        return PIERDE_CONTRA[ultima_ia]


class OponenteCopyBot(Oponente):
    """Copia la ultima jugada de la IA"""

    def decidir_jugada(self) -> str:
        if not self.historial:
            return np.random.choice(["piedra", "papel", "tijera"])

        return self.historial[-1][0]


class OponenteAntiCounterBot(Oponente):
    """Juega lo que gana a lo que le ganaria a su ultima jugada"""

    def decidir_jugada(self) -> str:
        if not self.historial:
            return np.random.choice(["piedra", "papel", "tijera"])

        ultima_propia = self.historial[-1][1]
        counter = PIERDE_CONTRA[ultima_propia]
        return PIERDE_CONTRA[counter]


class OponenteSesgoFuerte(Oponente):
    """Juega una opcion el 70% del tiempo"""

    def __init__(self, opcion_favorita="piedra"):
        super().__init__()
        self.opcion_favorita = opcion_favorita

    def decidir_jugada(self) -> str:
        if np.random.random() < 0.7:
            return self.opcion_favorita
        return np.random.choice(["piedra", "papel", "tijera"])


class OponenteSesgo80Piedra(Oponente):
    """80% piedra, 20% aleatorio"""

    def decidir_jugada(self) -> str:
        if np.random.random() < 0.8:
            return "piedra"
        return np.random.choice(["piedra", "papel", "tijera"])


class OponenteSesgo80Papel(Oponente):
    """80% papel, 20% aleatorio"""

    def decidir_jugada(self) -> str:
        if np.random.random() < 0.8:
            return "papel"
        return np.random.choice(["piedra", "papel", "tijera"])


class OponenteSesgo80Tijera(Oponente):
    """80% tijera, 20% aleatorio"""

    def decidir_jugada(self) -> str:
        if np.random.random() < 0.8:
            return "tijera"
        return np.random.choice(["piedra", "papel", "tijera"])


class OponenteVictor(Oponente):
    """
    Patron especifico de Victor:
    - 47% piedra (casi la mitad)
    - Despues de tijera → casi siempre piedra (90%)
    - Papel solo 22% (evita papel)
    - Despues de empates → piedra 56%
    - Rachas de piedra: si saco piedra 2 veces, 70% tercera piedra
    """

    def decidir_jugada(self) -> str:
        # Si hay historial, aplicar reglas de Victor
        if len(self.historial) >= 1:
            ultima_propia = self.historial[-1][1]

            # Regla 1: Despues de tijera → 90% piedra
            if ultima_propia == "tijera":
                if np.random.random() < 0.9:
                    return "piedra"

            # Regla 2: Despues de empate → 56% piedra
            ultima_ia = self.historial[-1][0]
            if ultima_ia == ultima_propia:
                if np.random.random() < 0.56:
                    return "piedra"

            # Regla 3: Rachas de piedra (si las ultimas 2 son piedra → 70% piedra)
            if len(self.historial) >= 2:
                ultimas_2 = [j2 for _, j2 in self.historial[-2:]]
                if ultimas_2[0] == "piedra" and ultimas_2[1] == "piedra":
                    if np.random.random() < 0.7:
                        return "piedra"

        # Distribucion general: 47% piedra, 31% tijera, 22% papel
        rand = np.random.random()
        if rand < 0.47:
            return "piedra"
        elif rand < 0.47 + 0.31:
            return "tijera"
        else:
            return "papel"


class OponenteCambiaDepuesPerder(Oponente):
    """Siempre cambia de jugada despues de perder"""

    def decidir_jugada(self) -> str:
        if not self.historial:
            return np.random.choice(["piedra", "papel", "tijera"])

        ultima_ia, ultima_propia = self.historial[-1]

        if GANA_A[ultima_ia] == ultima_propia:
            # Perdio, cambiar a algo diferente
            opciones = ["piedra", "papel", "tijera"]
            opciones.remove(ultima_propia)
            return np.random.choice(opciones)

        # Gano o empato, puede repetir o cambiar
        return np.random.choice(["piedra", "papel", "tijera"])


def calcular_resultado_ronda(jugada_ia, jugada_oponente):
    """Devuelve 1 si gana IA, 0 empate, -1 si pierde"""
    if jugada_ia == jugada_oponente:
        return 0
    if GANA_A[jugada_ia] == jugada_oponente:
        return 1
    return -1


def jugar_partida(ia: JugadorIA, oponente: Oponente, num_rondas=100, verbose=False):
    """
    Juega una partida completa entre la IA y el oponente.

    Returns:
        (victorias_ia, empates, derrotas_ia)
    """
    victorias = 0
    empates = 0
    derrotas = 0

    for ronda in range(num_rondas):
        jugada_ia = ia.decidir_jugada()
        jugada_oponente = oponente.decidir_jugada()

        resultado = calcular_resultado_ronda(jugada_ia, jugada_oponente)

        if resultado == 1:
            victorias += 1
        elif resultado == 0:
            empates += 1
        else:
            derrotas += 1

        ia.registrar_ronda(jugada_ia, jugada_oponente)
        oponente.registrar_ronda(jugada_ia, jugada_oponente)

        if verbose and ronda % 20 == 0:
            print(f"  Ronda {ronda}: IA={jugada_ia}, Oponente={jugada_oponente}, Resultado={resultado}")

    return victorias, empates, derrotas


def evaluar_patron(nombre_patron, clase_oponente, num_partidas=20, rondas_por_partida=100):
    """
    Evalua el modelo contra un patron especifico multiples veces.
    """
    print(f"\n{'='*70}")
    print(f"Evaluando contra: {nombre_patron}")
    print(f"{'='*70}")

    winrates = []

    for i in range(num_partidas):
        ia = JugadorIA()

        if callable(clase_oponente):
            oponente = clase_oponente()
        else:
            oponente = clase_oponente

        victorias, empates, derrotas = jugar_partida(ia, oponente, rondas_por_partida)

        total = victorias + empates + derrotas
        winrate = (victorias / total) * 100 if total > 0 else 0
        winrates.append(winrate)

        if (i + 1) % 5 == 0:
            print(f"  Partidas completadas: {i + 1}/{num_partidas}")

    winrates = np.array(winrates)

    print(f"\nResultados para {nombre_patron}:")
    print(f"  Winrate medio:    {winrates.mean():.2f}%")
    print(f"  Desviacion std:   {winrates.std():.2f}%")
    print(f"  Winrate minimo:   {winrates.min():.2f}%")
    print(f"  Winrate maximo:   {winrates.max():.2f}%")

    return {
        'media': winrates.mean(),
        'std': winrates.std(),
        'min': winrates.min(),
        'max': winrates.max(),
        'winrates': winrates
    }


def main():
    """Ejecuta todos los tests"""
    print("="*70)
    print("   TEST DE PATRONES ADVERSARIOS - RPSAI")
    print("="*70)
    print(f"\nConfiguracion:")
    print(f"  Rondas por partida: {RONDAS_POR_PARTIDA}")
    print(f"  Repeticiones por patron: {REPETICIONES_POR_PATRON}")
    print(f"\nEvaluando {len([1 for _ in range(14)])} patrones diferentes...\n")

    patrones = [
        ("Aleatorio", OponenteAleatorio),
        ("Piedra Constante", OponentePiedraConstante),
        ("Papel Constante", OponentePapelConstante),
        ("Tijera Constante", OponenteTijeraConstante),
        ("Ciclo (P->Pa->T)", OponenteCiclo),
        ("Counter-Bot", OponenteCounterBot),
        ("Copy-Bot", OponenteCopyBot),
        ("Anti-Counter-Bot", OponenteAntiCounterBot),
        ("Sesgo Piedra 70%", lambda: OponenteSesgoFuerte("piedra")),
        ("Sesgo 80% Piedra", OponenteSesgo80Piedra),
        ("Sesgo 80% Papel", OponenteSesgo80Papel),
        ("Sesgo 80% Tijera", OponenteSesgo80Tijera),
        ("VICTOR (Patron Real)", OponenteVictor),
        ("Cambia Tras Perder", OponenteCambiaDepuesPerder)
    ]

    resultados_globales = {}

    for nombre, clase in patrones:
        resultado = evaluar_patron(nombre, clase,
                                  num_partidas=REPETICIONES_POR_PATRON,
                                  rondas_por_partida=RONDAS_POR_PARTIDA)
        resultados_globales[nombre] = resultado

    print("\n" + "="*70)
    print("RESUMEN GLOBAL")
    print("="*70)
    print(f"\n{'Patron':<25} {'Media':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*70)

    for nombre, stats in resultados_globales.items():
        print(f"{nombre:<25} {stats['media']:>6.2f}%   {stats['std']:>6.2f}%   "
              f"{stats['min']:>6.2f}%   {stats['max']:>6.2f}%")





if __name__ == "__main__":
    main()