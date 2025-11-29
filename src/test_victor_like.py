"""
RPSAI - Test automatico estilo Victor
=====================================

Este script simula partidas contra un oponente "Victor-like",
cuyas jugadas se generan a partir de los patrones aprendidos
del propio CSV de partidas (data/partidas.csv).

Uso:
    py .\src\test_victor_like.py
    py .\src\test_victor_like.py -p 30 -n 50
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# Agregar el directorio src al path para importar modelo.py
sys.path.insert(0, str(Path(__file__).parent))

from modelo import JugadorIA, cargar_datos, GANA_A, JUGADA_A_NUM  # reutilizamos lo ya hecho


JUGADAS = ["piedra", "papel", "tijera"]


def resultado_ia(jugada_ia: str, jugada_op: str) -> str:
    """
    Devuelve el resultado desde el punto de vista de la IA.
    """
    if jugada_ia == jugada_op:
        return "empate"
    elif GANA_A[jugada_ia] == jugada_op:
        return "victoria"
    else:
        return "derrota"


class VictorModel:
    """
    Modelo simple de comportamiento de Victor basado en el CSV.

    - Aprende:
        * Distribucion de la PRIMERA jugada de Victor (j2)
        * Transiciones j2_actual -> j2_siguiente (por partida)
    """

    def __init__(self):
        # Contadores
        self.initial_counts = Counter()
        self.transition_counts = {j: Counter() for j in JUGADAS}
        self._ajustar_desde_csv()

    def _ajustar_desde_csv(self):
        """
        Carga el CSV con cargar_datos() y construye:
        - initial_counts: frecuencia de la primera jugada de Victor por partida
        - transition_counts: frecuencia de transiciones j2 -> j2_siguiente
        """
        df = cargar_datos()  # usa RUTA_DATOS de modelo.py

        # Normalizar columnas relevantes y ordenar por partida/ronda
        df = df.copy()
        for col in ["jugada_j1", "jugada_j2"]:
            df[col] = df[col].astype(str).str.strip().str.lower()

        if "partida" in df.columns and "numero_ronda" in df.columns:
            df = df.sort_values(["partida", "numero_ronda"]).reset_index(drop=True)
            grouped = df.groupby("partida", sort=False)
        else:
            # Si por lo que sea no hay 'partida', tratamos to-do como una sola
            df = df.sort_values(["numero_ronda"]).reset_index(drop=True)
            grouped = [("unica", df)]

        for _, g in grouped:
            jugadas_victor = list(g["jugada_j2"])

            if not jugadas_victor:
                continue

            # Primera jugada de Victor en la partida
            primera = jugadas_victor[0]
            if primera in JUGADAS:
                self.initial_counts[primera] += 1

            # Transiciones j2[i] -> j2[i+1]
            for i in range(len(jugadas_victor) - 1):
                actual = jugadas_victor[i]
                siguiente = jugadas_victor[i + 1]
                if actual in JUGADAS and siguiente in JUGADAS:
                    self.transition_counts[actual][siguiente] += 1

        # Por si acaso, si initial_counts esta vacio (CSV raro), forzamos algo razonable
        if sum(self.initial_counts.values()) == 0:
            for j in JUGADAS:
                self.initial_counts[j] = 1

    def _muestrear_desde_counter(self, counter: Counter) -> str:
        """
        Muestrea una jugada de un Counter de frecuencias.
        Si el contador esta vacio, devuelve una jugada uniforme.
        """
        total = sum(counter.values())
        if total <= 0:
            return np.random.choice(JUGADAS)

        jugadas = list(counter.keys())
        pesos = np.array(list(counter.values()), dtype=float)
        pesos = pesos / pesos.sum()

        return np.random.choice(jugadas, p=pesos)

    def primera_jugada(self) -> str:
        """
        Genera una primera jugada estilo Victor (j2).
        """
        return self._muestrear_desde_counter(self.initial_counts)

    def siguiente_jugada(self, j2_prev: str | None) -> str:
        """
        Genera la siguiente jugada de Victor, condicionado a su jugada anterior.

        - Si hay jugada anterior: usa transition_counts[j2_prev]
        - Si no hay jugada anterior: usa initial_counts
        """
        if j2_prev is None or j2_prev not in self.transition_counts:
            return self.primera_jugada()

        counter = self.transition_counts.get(j2_prev, None)
        if counter is None or sum(counter.values()) == 0:
            return self.primera_jugada()

        return self._muestrear_desde_counter(counter)


class EvaluadorVictorLike:
    """
    Evalua la IA contra un oponente estilo Victor (modelo de Markov sencillo).
    """

    def __init__(self, num_rondas: int = 50, num_partidas: int = 30):
        """
        Args:
            num_rondas: numero de rondas por partida (ej: 50)
            num_partidas: cuantas partidas simular
        """
        self.num_rondas = num_rondas
        self.num_partidas = num_partidas
        self.victor_model = VictorModel()

    def simular(self):
        """
        Ejecuta las simulaciones y muestra un resumen de winrates.
        """
        print("=" * 60)
        print("   RPSAI - TEST ESTILO VICTOR (Simulacion)")
        print("=" * 60)
        print(f"Rondas por partida: {self.num_rondas}")
        print(f"Numero de partidas simuladas: {self.num_partidas}\n")

        winrates = []

        for pid in range(1, self.num_partidas + 1):
            ia = JugadorIA()  # carga el modelo entrenado
            historial_ia = []
            historial_op = []

            victorias = 0
            derrotas = 0
            empates = 0

            j2_prev = None  # ultima jugada de Victor en esta simulacion

            for ronda in range(1, self.num_rondas + 1):
                # IA decide
                jugada_ia = ia.decidir_jugada()

                # Victor-like decide en base a su propio historial (Markov1)
                if ronda == 1:
                    jugada_victor = self.victor_model.primera_jugada()
                else:
                    jugada_victor = self.victor_model.siguiente_jugada(j2_prev)

                j2_prev = jugada_victor

                # Resultado para la IA
                res = resultado_ia(jugada_ia, jugada_victor)
                if res == "victoria":
                    victorias += 1
                elif res == "derrota":
                    derrotas += 1
                else:
                    empates += 1

                # MUY IMPORTANTE: registrar con IA como j1 y Victor-bot como j2
                ia.registrar_ronda(jugada_ia, jugada_victor)

                historial_ia.append(jugada_ia)
                historial_op.append(jugada_victor)

            total_decisivas = victorias + derrotas
            if total_decisivas > 0:
                winrate = victorias / total_decisivas * 100
            else:
                winrate = 0.0

            winrates.append(winrate)
            print(
                f"  Partida {pid:02d}: "
                f"{victorias}V-{derrotas}D-{empates}E | "
                f"Winrate IA: {winrate:.1f}%"
            )

        winrates = np.array(winrates, dtype=float)
        media = float(winrates.mean())
        std = float(winrates.std())
        w_min = float(winrates.min())
        w_max = float(winrates.max())

        print("\n" + "=" * 60)
        print("   RESUMEN GLOBAL VS VICTOR-LIKE")
        print("=" * 60)
        print(f"  Winrate medio IA: {media:.2f}%")
        print(f"  Desviacion std  : {std:.2f}")
        print(f"  Min / Max       : {w_min:.1f}% / {w_max:.1f}%")
        print("\n[FIN TEST VICTOR-LIKE]")


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Simula partidas contra un oponente estilo Victor usando el CSV real"
    )
    parser.add_argument(
        "-n",
        "--rondas",
        type=int,
        default=50,
        help="Rondas por partida (default: 50)",
    )
    parser.add_argument(
        "-p",
        "--partidas",
        type=int,
        default=30,
        help="Numero de partidas a simular (default: 30)",
    )
    args = parser.parse_args()

    evaluador = EvaluadorVictorLike(
        num_rondas=args.rondas,
        num_partidas=args.partidas,
    )
    evaluador.simular()


if __name__ == "__main__":
    main()
