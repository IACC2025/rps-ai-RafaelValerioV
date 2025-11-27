"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================

INSTRUCCIONES PARA EL ALUMNO:
-----------------------------
Este archivo contiene la plantilla para tu modelo de IA.
Debes completar las secciones marcadas con TODO.

El objetivo es crear un modelo que prediga la PROXIMA jugada del oponente
y responda con la jugada que le gana.

FORMATO DEL CSV (minimo requerido):
-----------------------------------
Tu archivo data/partidas.csv debe tener AL MENOS estas columnas:
    - numero_ronda: Numero de la ronda (1, 2, 3...)
    - jugada_j1: Jugada del jugador 1 (piedra/papel/tijera)
    - jugada_j2: Jugada del jugador 2/oponente (piedra/papel/tijera)

Ejemplo:
    numero_ronda,jugada_j1,jugada_j2
    1,piedra,papel
    2,tijera,piedra
    3,papel,papel

Si has capturado datos adicionales (tiempo_reaccion, timestamp, etc.),
puedes usarlos para crear features extra.

EVALUACION:
- 30% Extraccion de datos (documentado en DATOS.md)
- 30% Feature Engineering
- 40% Entrenamiento y funcionamiento del modelo

FLUJO:
1. Cargar datos del CSV
2. Crear features (caracteristicas predictivas)
3. Entrenar modelo(s)
4. Evaluar y seleccionar el mejor
5. Usar el modelo para predecir y jugar
"""

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Descomenta esta linea si te molesta el warning de sklearn sobre feature names:
# warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas a numeros (para el modelo)
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


# =============================================================================
# PARTE 1: EXTRACCION DE DATOS (30% de la nota)
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga los datos del CSV de partidas.

    - Lee el CSV con pandas
    - Normaliza nombres de columnas a:
      partida, numero_ronda, jugada_j1 (humano), jugada_j2 (IA)
    """
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    ruta_csv = Path(ruta_csv)

    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    # Normalizar nombre de columna de ronda
    if "numero_ronda" not in df.columns and "ronda" in df.columns:
        df = df.rename(columns={"ronda": "numero_ronda"})

    # Adaptar tu CSV: columnas rafa/victor -> jugada_j1/jugada_j2
    # Consideramos que 'victor' es el humano (jugador 1) y 'rafa' el jugador 2
    if "victor" in df.columns and "rafa" in df.columns:
        df["jugada_j1"] = df["victor"].astype(str).str.lower().str.strip()
        df["jugada_j2"] = df["rafa"].astype(str).str.lower().str.strip()
    else:
        # Caso general si ya viene con jugada_j1/jugada_j2
        if not {"jugada_j1", "jugada_j2"}.issubset(df.columns):
            raise ValueError(
                "El CSV debe tener columnas 'victor' y 'rafa' "
                "o bien 'jugada_j1' y 'jugada_j2'."
            )

    # Ordenar por partida y numero de ronda
    df = df.sort_values(["partida", "numero_ronda"]).reset_index(drop=True)

    return df



def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelo.

    - Convierte jugadas texto -> números (0 piedra, 1 papel, 2 tijera)
    - Crea la columna 'proxima_jugada_j2' como:
        la siguiente jugada del HUMANO (jugada_j1) dentro de la misma partida
    - Elimina filas sin 'proxima_jugada_j2' (última ronda de cada partida)
    """
    df = df.copy()

    # Convertir jugadas a números
    df["jugada_j1_num"] = df["jugada_j1"].map(JUGADA_A_NUM)
    df["jugada_j2_num"] = df["jugada_j2"].map(JUGADA_A_NUM)

    if df["jugada_j1_num"].isna().any() or df["jugada_j2_num"].isna().any():
        raise ValueError("Hay jugadas no válidas en el CSV (que no son piedra/papel/tijera)")

    # Target: próxima jugada DEL HUMANO (jugada_j1) en la misma partida
    df["proxima_jugada_j2"] = (
        df.groupby("partida")["jugada_j1_num"].shift(-1)
    )

    # Eliminar filas sin target (última ronda de cada partida)
    df = df.dropna(subset=["proxima_jugada_j2"]).reset_index(drop=True)
    df["proxima_jugada_j2"] = df["proxima_jugada_j2"].astype(int)

    return df



# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features (características) para el modelo.

    Features principales:
    - num_rondas_jugadas
    - freqs del humano: freq_j1_piedra, freq_j1_papel, freq_j1_tijera
    - últimas jugadas (lags) de humano e IA
    - resultado_j1 y resultado_anterior (1 gana, 0 empate, -1 pierde)
    """
    df = df.copy()

    # Asegurarnos de estar ordenados
    df = df.sort_values(["partida", "numero_ronda"]).reset_index(drop=True)

    # Número de rondas jugadas hasta ese momento en la partida
    df["num_rondas_jugadas"] = df.groupby("partida").cumcount() + 1

    # Frecuencias acumuladas del humano (jugada_j1)
    for mov in ["piedra", "papel", "tijera"]:
        indicador = (df["jugada_j1"] == mov).astype(int)
        df[f"count_j1_{mov}"] = indicador.groupby(df["partida"]).cumsum()
        df[f"freq_j1_{mov}"] = df[f"count_j1_{mov}"] / df["num_rondas_jugadas"]

    # Lag features (jugadas anteriores) para humano e IA
    grp_j1 = df.groupby("partida")["jugada_j1_num"]
    grp_j2 = df.groupby("partida")["jugada_j2_num"]

    # Última jugada (equivale a jugada_j1_num/jugada_j2_num del propio df)
    df["j1_ult"] = grp_j1.shift(0)
    df["j1_lag1"] = grp_j1.shift(1)
    df["j1_lag2"] = grp_j1.shift(2)

    df["j2_ult"] = grp_j2.shift(0)
    df["j2_lag1"] = grp_j2.shift(1)
    df["j2_lag2"] = grp_j2.shift(2)

    # Resultado desde el punto de vista del humano (jugada_j1)
    def resultado_fila(row):
        j1 = row["jugada_j1"]
        j2 = row["jugada_j2"]
        if j1 == j2:
            return 0
        elif GANA_A[j1] == j2:
            return 1
        else:
            return -1

    df["resultado_j1"] = df.apply(resultado_fila, axis=1)
    grp_res = df.groupby("partida")["resultado_j1"]
    df["resultado_anterior"] = grp_res.shift(1)

    # Rellenar NaN de lags y resultado_anterior con 0 (valor neutro)
    lag_cols = ["j1_lag1", "j1_lag2", "j2_lag1", "j2_lag2", "resultado_anterior"]
    df[lag_cols] = df[lag_cols].fillna(0)

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.

    X: matriz de características
    y: próxima jugada del humano (proxima_jugada_j2)
    """
    df = df.copy()

    # Por si acaso
    df = df.dropna(subset=["proxima_jugada_j2"])

    feature_cols = [
        "num_rondas_jugadas",
        "jugada_j1_num",
        "jugada_j2_num",
        "j1_lag1",
        "j1_lag2",
        "j2_lag1",
        "j2_lag2",
        "freq_j1_piedra",
        "freq_j1_papel",
        "freq_j1_tijera",
        "resultado_j1",
        "resultado_anterior",
    ]

    X = df[feature_cols].values
    y = df["proxima_jugada_j2"].values

    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de predicción.

    - Divide en train/test
    - Entrena KNN y RandomForest
    - Muestra métricas
    - Devuelve el mejor modelo reentrenado con todos los datos
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    modelos = {
        "KNN_5": KNeighborsClassifier(n_neighbors=5),
        "KNN_7": KNeighborsClassifier(n_neighbors=7),
        "KNN_9": KNeighborsClassifier(n_neighbors=9),
        "KNN_11": KNeighborsClassifier(n_neighbors=11),
        "RandomForest_depth2": RandomForestClassifier(
            n_estimators=300,
            max_depth=2,
            random_state=42
        ),
        "RandomForest_depth3": RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            random_state=42
        ),
    }

    mejor_modelo = None
    mejor_nombre = None
    mejor_acc = -1.0

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("\n==============================")
        print(f"Modelo: {nombre}")
        print(f"Accuracy en test: {acc:.3f}")
        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))
        print("\nInforme de clasificación:")
        print(classification_report(y_test, y_pred, digits=3))

        if acc > mejor_acc:
            mejor_acc = acc
            mejor_nombre = nombre
            mejor_modelo = modelo

    print("\n==============================")
    print(f"Mejor modelo: {mejor_nombre} (accuracy={mejor_acc:.3f})")
    print("Reentrenando mejor modelo con TODOS los datos...")

    # Reentrenar con todos los datos
    mejor_modelo.fit(X, y)

    return mejor_modelo



def guardar_modelo(modelo, ruta: str = None):
    """Guarda el modelo entrenado en un archivo."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")

    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# PARTE 4: PREDICCION Y JUEGO
# =============================================================================

class JugadorIA:
    """
    Clase que encapsula el modelo para jugar.

    - Carga un modelo entrenado desde disco
    - Mantiene un historial de rondas jugadas (humano, IA)
    - A partir del historial calcula las MISMAS features que usaste al entrenar
    - Predice la próxima jugada del oponente y juega lo que le gana
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []  # Lista de tuplas (jugada_j1, jugada_j2)

        if ruta_modelo is None:
            ruta_modelo = RUTA_MODELO

        # Intentar cargar el modelo entrenado
        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print(f"[INFO] JugadorIA: modelo cargado desde {ruta_modelo}")
        except FileNotFoundError:
            print("[ADVERTENCIA] JugadorIA: no se encontró un modelo entrenado.")
            print("              La IA jugará de forma ALEATORIA "
                  "hasta que entrenes y guardes el modelo.")
            self.modelo = None
        except Exception as e:
            print(f"[ADVERTENCIA] JugadorIA: error al cargar el modelo: {e}")
            print("              La IA jugará de forma ALEATORIA.")
            self.modelo = None

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del humano (oponente)
            jugada_j2: Jugada de la IA
        """
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> np.ndarray | None:
        """
        Genera las features basadas en el historial actual.

        IMPORTANTE:
        - Deben ser las MISMAS features y en el MISMO ORDEN
          que usaste en seleccionar_features() para entrenar.
        - Cada elemento del historial es una ronda ya jugada.

        Returns:
            Array 1D con las features para la predicción
            o None si no hay suficiente historial.
        """
        # Si no hay historial, no podemos calcular nada razonable
        if len(self.historial) == 0:
            return None

        # Construimos un DataFrame pequeño con la partida actual
        # Consideramos toda la sesión como una sola partida (id = 1)
        filas = []
        for i, (j1, j2) in enumerate(self.historial, start=1):
            # Si alguna jugada no es válida, devolvemos None
            if j1 not in JUGADA_A_NUM or j2 not in JUGADA_A_NUM:
                return None

            filas.append({
                "partida": 1,
                "numero_ronda": i,
                "jugada_j1": j1,
                "jugada_j2": j2,
                "jugada_j1_num": JUGADA_A_NUM[j1],
                "jugada_j2_num": JUGADA_A_NUM[j2],
            })

        df = pd.DataFrame(filas)

        # Número de rondas jugadas hasta ese momento
        df["num_rondas_jugadas"] = df["numero_ronda"]

        # Frecuencias acumuladas de jugadas del humano (jugada_j1)
        for mov in ["piedra", "papel", "tijera"]:
            indicador = (df["jugada_j1"] == mov).astype(int)
            df[f"count_j1_{mov}"] = indicador.cumsum()
            df[f"freq_j1_{mov}"] = df[f"count_j1_{mov}"] / df["num_rondas_jugadas"]

        # Lag features (jugadas anteriores) de humano e IA
        grp_j1 = df["jugada_j1_num"]
        grp_j2 = df["jugada_j2_num"]

        df["j1_lag1"] = grp_j1.shift(1)
        df["j1_lag2"] = grp_j1.shift(2)
        df["j2_lag1"] = grp_j2.shift(1)
        df["j2_lag2"] = grp_j2.shift(2)

        # Resultado de cada ronda visto desde el humano (jugada_j1)
        def resultado_fila(row):
            j1 = row["jugada_j1"]
            j2 = row["jugada_j2"]
            if j1 == j2:
                return 0
            elif GANA_A[j1] == j2:
                return 1   # humano gana
            else:
                return -1  # humano pierde

        df["resultado_j1"] = df.apply(resultado_fila, axis=1)
        df["resultado_anterior"] = df["resultado_j1"].shift(1)

        # Rellenar NaN de lags y resultado_anterior con 0 (valor neutro)
        lag_cols = ["j1_lag1", "j1_lag2", "j2_lag1", "j2_lag2", "resultado_anterior"]
        df[lag_cols] = df[lag_cols].fillna(0)

        # Tomamos SOLO la última fila (estado actual de la partida)
        ultima = df.iloc[-1]

        # Mismo orden de columnas que en seleccionar_features()
        feature_cols = [
            "num_rondas_jugadas",
            "jugada_j1_num",
            "jugada_j2_num",
            "j1_lag1",
            "j1_lag2",
            "j2_lag1",
            "j2_lag2",
            "freq_j1_piedra",
            "freq_j1_papel",
            "freq_j1_tijera",
            "resultado_j1",
            "resultado_anterior",
        ]

        features = ultima[feature_cols].to_numpy(dtype=float)
        return features

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la próxima jugada del oponente (humano).

        - Si no hay modelo o no hay historial, juega aleatorio.
        """
        # Sin modelo -> aleatorio
        if self.modelo is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        features = self.obtener_features_actuales()

        # Si no podemos construir features (por falta de historial, etc.)
        if features is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        # Predicción del modelo (0/1/2)
        pred_num = self.modelo.predict(features.reshape(1, -1))[0]

        # Convertimos número -> jugada en texto
        jugada = NUM_A_JUGADA.get(int(pred_num), "piedra")
        return jugada

    def decidir_jugada(self) -> str:
        """
        Decide qué jugada hacer para ganar al oponente.

        - Predice la jugada del humano
        - Devuelve la jugada que le gana
        """
        prediccion_oponente = self.predecir_jugada_oponente()

        if prediccion_oponente is None:
            # Por seguridad, pero en principio nunca debería ser None aquí
            return np.random.choice(["piedra", "papel", "tijera"])

        # PIERDE_CONTRA[x] es la jugada que gana a x
        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """
    Función principal para entrenar el modelo.

    Ejecuta: python src/modelo.py
    """
    print("="*50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*50)

    # 1. Cargar datos
    df = cargar_datos()
    print(f"[INFO] Datos cargados: {len(df)} filas")

    # 2. Preparar datos (target, codificación numérica)
    df_prep = preparar_datos(df)
    print(f"[INFO] Datos con target: {len(df_prep)} filas")

    # 3. Crear features
    df_feat = crear_features(df_prep)

    # 4. Seleccionar X, y
    X, y = seleccionar_features(df_feat)
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    # 5. Entrenar modelo
    modelo = entrenar_modelo(X, y)

    # 6. Guardar modelo
    guardar_modelo(modelo)



if __name__ == "__main__":
    main()
