"""
RPSAI - Test automatico de patrones
===================================

Este script evalua el comportamiento de la IA contra varios patrones
clasicos de oponente (constante, ciclo, copy-bot, counter-bot, etc.)
sin intervencion humana.

Uso:
    py .\src\test_patrones.py
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio src al path para importar modelo.py
sys.path.insert(0, str(Path(__file__).parent))

from modelo import JugadorIA, GANA_A

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


class EvaluadorPatrones:
    """
    Clase para evaluar automaticamente la IA contra distintos patrones.
    """

    def __init__(self, num_rondas: int = 50, num_evals: int = 30):
        """
        Args:
            num_rondas: rondas por partida (ej: 50, como en la evaluacion real)
            num_evals: cuantas veces repetir cada test para promediar
        """
        self.num_rondas = num_rondas
        self.num_evals = num_evals

    # ----------------- PATRONES DE OPONENTE -----------------

    def _estrategia_constante(self, jugada_constante: str):
        def fn(ronda, hist_ia, hist_op):
            return jugada_constante
        return fn

    def _estrategia_alternancia(self, j1: str, j2: str):
        secuencia = [j1, j2]

        def fn(ronda, hist_ia, hist_op):
            idx = (ronda - 1) % 2
            return secuencia[idx]
        return fn

    def _estrategia_ciclo3(self, s0: str, s1: str, s2: str):
        secuencia = [s0, s1, s2]

        def fn(ronda, hist_ia, hist_op):
            idx = (ronda - 1) % 3
            return secuencia[idx]
        return fn

    def _estrategia_ciclo_largo(self, secuencia):
        L = len(secuencia)

        def fn(ronda, hist_ia, hist_op):
            idx = (ronda - 1) % L
            return secuencia[idx]
        return fn

    def _estrategia_copy_bot(self, ronda, hist_ia, hist_op):
        """
        Copia SIEMPRE la jugada anterior de la IA.
        """
        if not hist_ia:
            return "piedra"  # primera jugada arbitraria
        return hist_ia[-1]

    def _estrategia_counter_bot(self, ronda, hist_ia, hist_op):
        """
        Juega SIEMPRE lo que gana a la jugada anterior de la IA.
        """
        from modelo import PIERDE_CONTRA  # evitar import circular arriba

        if not hist_ia:
            base = "piedra"
        else:
            base = hist_ia[-1]
        return PIERDE_CONTRA[base]

    def _estrategia_cambio_mitad(self):
        """
        Mitad 1: oponente random puro.
        Mitad 2: ciclo largo fijo.
        """
        mitad = self.num_rondas // 2
        ciclo = ["piedra", "papel", "papel", "tijera", "papel"]
        L = len(ciclo)

        def fn(ronda, hist_ia, hist_op):
            if ronda <= mitad:
                return np.random.choice(JUGADAS)
            else:
                idx = (ronda - mitad - 1) % L
                return ciclo[idx]

        return fn

    # ----------------- MOTOR DE SIMULACION -----------------

    def _simular_patron(self, nombre: str, estrategia_op):
        """
        Simula varias partidas contra una estrategia de oponente dada.

        Args:
            nombre: nombre del patron (solo para imprimir)
            estrategia_op: funcion (ronda, hist_ia, hist_op) -> jugada_oponente

        Returns:
            lista de winrates (por partida) de la IA
        """
        print(f"\n=== Patron: {nombre} ===")

        winrates = []

        for eval_id in range(1, self.num_evals + 1):
            ia = JugadorIA()  # carga el modelo entrenado
            historial_ia = []
            historial_op = []

            victorias = 0
            derrotas = 0
            empates = 0

            for ronda in range(1, self.num_rondas + 1):
                # IA decide su jugada
                jugada_ia = ia.decidir_jugada()

                # Oponente juega segun la estrategia
                jugada_op = estrategia_op(ronda, historial_ia, historial_op)

                # Resultado y actualizacion de historial
                res = resultado_ia(jugada_ia, jugada_op)

                if res == "victoria":
                    victorias += 1
                elif res == "derrota":
                    derrotas += 1
                else:
                    empates += 1

                # IMPORTANTE: registrar con IA como j1 y oponente como j2
                ia.registrar_ronda(jugada_ia, jugada_op)

                historial_ia.append(jugada_ia)
                historial_op.append(jugada_op)

            total_decisivas = victorias + derrotas
            if total_decisivas > 0:
                winrate = victorias / total_decisivas * 100
            else:
                winrate = 0.0

            winrates.append(winrate)
            print(
                f"  Eval {eval_id:02d}: "
                f"{victorias}V-{derrotas}D-{empates}E | "
                f"Winrate IA: {winrate:.1f}%"
            )

        winrates = np.array(winrates)
        media = winrates.mean()
        std = winrates.std()
        w_min = winrates.min()
        w_max = winrates.max()

        print(f"\n[Resumen patron: {nombre}]")
        print(f"  Winrate medio IA: {media:.2f}%")
        print(f"  Desviacion std  : {std:.2f}")
        print(f"  Min / Max       : {w_min:.1f}% / {w_max:.1f}%")

        return winrates

    # ----------------- EJECUTAR TODOS LOS TESTS -----------------

    def ejecutar_todos(self):
        """
        Ejecuta los tests para todos los patrones definidos.
        """
        print("=" * 60)
        print("   RPSAI - TEST AUTOMATICO DE PATRONES")
        print("=" * 60)
        print(f"Rondas por partida: {self.num_rondas}")
        print(f"Repeticiones por patron: {self.num_evals}\n")

        resultados = {}

        # 1) Constantes
        for jug in JUGADAS:
            nombre = f"Constante ({jug})"
            estrategia = self._estrategia_constante(jug)
            resultados[nombre] = self._simular_patron(nombre, estrategia)

        # 2) Alternancia simple
        resultados["Alternancia (piedra/papel)"] = self._simular_patron(
            "Alternancia (piedra/papel)",
            self._estrategia_alternancia("piedra", "papel"),
        )
        resultados["Alternancia (piedra/tijera)"] = self._simular_patron(
            "Alternancia (piedra/tijera)",
            self._estrategia_alternancia("piedra", "tijera"),
        )

        # 3) Ciclo de 3
        resultados["Ciclo3 (P/A/T)"] = self._simular_patron(
            "Ciclo3 (piedra/papel/tijera)",
            self._estrategia_ciclo3("piedra", "papel", "tijera"),
        )

        # 4) Copy-bot
        resultados["Copy-bot"] = self._simular_patron(
            "Copy-bot",
            self._estrategia_copy_bot,
        )

        # 5) Counter-bot
        resultados["Counter-bot"] = self._simular_patron(
            "Counter-bot",
            self._estrategia_counter_bot,
        )

        # 6) Ciclo largo tipo [1,2,2,3,2] -> [P, A, A, T, A]
        ciclo_largo = ["piedra", "papel", "papel", "tijera", "papel"]
        resultados["Ciclo largo (P,A,A,T,A)"] = self._simular_patron(
            "Ciclo largo (P,A,A,T,A)",
            self._estrategia_ciclo_largo(ciclo_largo),
        )

        # 7) Cambio de estrategia a mitad
        resultados["Cambio mitad (random -> ciclo)"] = self._simular_patron(
            "Cambio mitad (random -> ciclo largo)",
            self._estrategia_cambio_mitad(),
        )

        print("\n" + "=" * 60)
        print("   RESUMEN GLOBAL DE PATRONES")
        print("=" * 60)

        for nombre, winrates in resultados.items():
            media = float(np.mean(winrates))
            std = float(np.std(winrates))
            w_min = float(np.min(winrates))
            w_max = float(np.max(winrates))

            # Una sola linea con toda la info
            print(
                f"  {nombre:35s} -> "
                f"media: {media:6.2f}% | "
                f"std: {std:5.2f} | "
                f"min/max: {w_min:5.1f}% / {w_max:5.1f}%"
            )

        print("\n[FIN TEST PATRONES]")


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Test automatico de patrones para la IA RPSAI")
    parser.add_argument(
        "-n",
        "--rondas",
        type=int,
        default=50,
        help="Rondas por partida (default: 50)",
    )
    parser.add_argument(
        "-e",
        "--evals",
        type=int,
        default=30,
        help="Numero de partidas por patron (default: 30)",
    )
    args = parser.parse_args()

    evaluador = EvaluadorPatrones(num_rondas=args.rondas, num_evals=args.evals)
    evaluador.ejecutar_todos()


if __name__ == "__main__":
    main()
