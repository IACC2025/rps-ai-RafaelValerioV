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
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

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

# Listas útiles para features
JUGADAS = ["piedra", "papel", "tijera"]
RESULTADOS = ["empate", "j1_gana", "j2_gana"]

# Columnas de features que usaremos para entrenar y en JugadorIA
FEATURE_COLS = [
    # Estado inmediato: última jugada de j2 (one-hot)
    "j2_ult_piedra",
    "j2_ult_papel",
    "j2_ult_tijera",
    # Estado inmediato: última jugada de j1 (one-hot)
    "j1_ult_piedra",
    "j1_ult_papel",
    "j1_ult_tijera",
    # Resultado de la ronda anterior (one-hot)
    "resultado_prev_empate",
    "resultado_prev_j1_gana",
    "resultado_prev_j2_gana",
    # Frecuencias j2 en ventana corta (últimas 3)
    "freq_j2_piedra_ult3",
    "freq_j2_papel_ult3",
    "freq_j2_tijera_ult3",
    # Frecuencias j2 en ventana media (últimas 10)
    "freq_j2_piedra_ult10",
    "freq_j2_papel_ult10",
    "freq_j2_tijera_ult10",
    # Racha de j2 repitiendo la misma mano
    "racha_j2_misma_mano",
    # Proporción de cambios de jugada de j2 en últimas 5
    "prop_cambios_j2_ult5",
    # Tasa de comportamiento tipo copy-bot y counter-bot
    "copy_rate_ult10",
    "counter_rate_ult10",
    # Entropía de las jugadas de j2 (nivel de aleatoriedad)
    "entropia_j2_ult10",
]


# -------------------------------------------------------------------------
# Pequeño helper: resultado de una ronda desde el punto de vista de j1/j2
# -------------------------------------------------------------------------
def resultado_ronda(j1: str, j2: str) -> str:
    """
    Devuelve:
        - 'empate'   si j1 == j2
        - 'j1_gana'  si j1 gana a j2
        - 'j2_gana'  en caso contrario
    """
    if pd.isna(j1) or pd.isna(j2):
        return np.nan
    if j1 == j2:
        return "empate"
    elif GANA_A.get(j1) == j2:
        return "j1_gana"
    else:
        return "j2_gana"


# =============================================================================
# PARTE 1: EXTRACCION DE DATOS (30% de la nota)
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga los datos del CSV de partidas.

    - Usa pandas para leer el CSV
    - Maneja el caso de que el archivo no exista
    - Verifica que tenga las columnas necesarias

    Args:
        ruta_csv: Ruta al archivo CSV (usa RUTA_DATOS por defecto)

    Returns:
        DataFrame con los datos de las partidas
    """
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    ruta_csv = Path(ruta_csv)

    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos en: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    columnas_minimas = ["numero_ronda", "jugada_j1", "jugada_j2"]
    faltan = [c for c in columnas_minimas if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {faltan}")

    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelo.

    - Convierte las jugadas de texto a numeros
    - Crea la columna 'proxima_jugada_j2' (el target a predecir)
    - Elimina filas con valores nulos

    Importante:
    - La fila t representa la ronda t.
    - El target es la jugada_j2 en la ronda t+1, dentro de la MISMA partida.

    Args:
        df: DataFrame con los datos crudos

    Returns:
        DataFrame preparado para feature engineering
    """
    df = df.copy()

    # Normalizar texto de jugadas
    for col in ["jugada_j1", "jugada_j2"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Ordenar por partida y numero_ronda si existe 'partida'
    if "partida" in df.columns:
        df = df.sort_values(["partida", "numero_ronda"]).reset_index(drop=True)
    else:
        df = df.sort_values(["numero_ronda"]).reset_index(drop=True)

    # Convertir jugadas a numeros
    df["jugada_j1_num"] = df["jugada_j1"].map(JUGADA_A_NUM)
    df["jugada_j2_num"] = df["jugada_j2"].map(JUGADA_A_NUM)

    # Crear target: proxima jugada de j2 (numerica)
    if "partida" in df.columns:
        df["proxima_jugada_j2"] = (
            df.groupby("partida")["jugada_j2_num"].shift(-1)
        )
    else:
        df["proxima_jugada_j2"] = df["jugada_j2_num"].shift(-1)

    # Eliminar filas sin target o sin mapeo de jugada
    df = df.dropna(
        subset=["jugada_j1_num", "jugada_j2_num", "proxima_jugada_j2"]
    ).reset_index(drop=True)

    # Asegurar tipo entero para el modelo
    df["jugada_j1_num"] = df["jugada_j1_num"].astype(int)
    df["jugada_j2_num"] = df["jugada_j2_num"].astype(int)
    df["proxima_jugada_j2"] = df["proxima_jugada_j2"].astype(int)

    return df


# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features (caracteristicas) para el modelo.

    Implementa el conjunto minimo de features diseñado en la FASE 1:

    1. Estado inmediato:
       - j2_ult_jugada (one-hot)
       - j1_ult_jugada (one-hot)
       - resultado de la ronda anterior (one-hot)

    2. Memoria corta y media de j2:
       - Frecuencia de cada jugada en ultimas 3 rondas
       - Frecuencia de cada jugada en ultimas 10 rondas
       - Racha actual repetida de j2
       - Proporcion de cambios j2 en ultimas 5

    3. Meta-comportamiento:
       - copy_rate_ult10: cuanto copia j2 tu jugada anterior
       - counter_rate_ult10: cuanto juega j2 lo que gana a tu jugada anterior
       - entropia_j2_ult10: aleatoriedad de j2 en ultimas 10

    Args:
        df: DataFrame con datos preparados (incluye proxima_jugada_j2)

    Returns:
        DataFrame con todas las features creadas
    """
    df = df.copy()

    # Asegurar orden y clave de grupo
    if "partida" in df.columns and "numero_ronda" in df.columns:
        df = df.sort_values(["partida", "numero_ronda"]).reset_index(drop=True)
        group_key = df["partida"]
    elif "partida" in df.columns:
        df = df.sort_values(["partida"]).reset_index(drop=True)
        group_key = df["partida"]
    else:
        df = df.reset_index(drop=True)
        group_key = None

    # ------------------------------------------------------------------
    # Feature 1: Estado inmediato - ultima jugada de j2 y j1 (one-hot)
    # ------------------------------------------------------------------
    for jug in JUGADAS:
        df[f"j2_ult_{jug}"] = (df["jugada_j2"] == jug).astype(int)
        df[f"j1_ult_{jug}"] = (df["jugada_j1"] == jug).astype(int)

    # ------------------------------------------------------------------
    # Feature 2: Resultado anterior (one-hot)
    # ------------------------------------------------------------------
    # Resultado actual
    df["resultado"] = [
        resultado_ronda(j1, j2)
        for j1, j2 in zip(df["jugada_j1"], df["jugada_j2"])
    ]

    # Resultado previo
    if group_key is not None:
        df["resultado_prev"] = df.groupby(group_key)["resultado"].shift(1)
    else:
        df["resultado_prev"] = df["resultado"].shift(1)

    res_dummies = pd.get_dummies(df["resultado_prev"], prefix="resultado_prev")
    for res in RESULTADOS:
        col = f"resultado_prev_{res}"
        if col in res_dummies.columns:
            df[col] = res_dummies[col]
        else:
            df[col] = 0

    # ------------------------------------------------------------------
    # Helpers para rolling por grupo
    # ------------------------------------------------------------------
    def rolling_mean_by_group(series: pd.Series, window: int) -> pd.Series:
        if group_key is not None:
            return (
                series.groupby(group_key)
                .apply(lambda s: s.rolling(window=window, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
        else:
            return series.rolling(window=window, min_periods=1).mean()

    # ------------------------------------------------------------------
    # Feature 3: Frecuencias de jugadas de j2 (ventanas 3 y 10)
    # ------------------------------------------------------------------
    for jug in JUGADAS:
        base = (df["jugada_j2"] == jug).astype(int)
        df[f"freq_j2_{jug}_ult3"] = rolling_mean_by_group(base, 3)
        df[f"freq_j2_{jug}_ult10"] = rolling_mean_by_group(base, 10)

    # ------------------------------------------------------------------
    # Feature 4: Racha de j2 repitiendo la misma mano
    # ------------------------------------------------------------------
    streak_values = []
    if group_key is not None:
        for _, g in df.groupby(group_key, sort=False):
            last = None
            c = 0
            for v in g["jugada_j2"]:
                if v == last:
                    c += 1
                else:
                    c = 1
                    last = v
                streak_values.append(c)
        df["racha_j2_misma_mano"] = streak_values
    else:
        last = None
        c = 0
        for v in df["jugada_j2"]:
            if v == last:
                c += 1
            else:
                c = 1
                last = v
            streak_values.append(c)
        df["racha_j2_misma_mano"] = streak_values

    # ------------------------------------------------------------------
    # Feature 5: Proporcion de cambios de jugada de j2 en ultimas 5 rondas
    # ------------------------------------------------------------------
    if group_key is not None:
        cambio = (
            df["jugada_j2"] != df.groupby(group_key)["jugada_j2"].shift(1)
        ).astype(int)
    else:
        cambio = (df["jugada_j2"] != df["jugada_j2"].shift(1)).astype(int)

    df["prop_cambios_j2_ult5"] = rolling_mean_by_group(cambio, 5)

    # ------------------------------------------------------------------
    # Feature 6: Copy rate y counter rate (ventana 10)
    # ------------------------------------------------------------------
    if group_key is not None:
        df["jugada_j1_prev"] = df.groupby(group_key)["jugada_j1"].shift(1)
    else:
        df["jugada_j1_prev"] = df["jugada_j1"].shift(1)

    df["es_copia"] = (df["jugada_j2"] == df["jugada_j1_prev"]).astype(int)

    df["counter_esperado"] = df["jugada_j1_prev"].map(PIERDE_CONTRA)
    df["es_counter"] = (df["jugada_j2"] == df["counter_esperado"]).astype(int)

    df["copy_rate_ult10"] = rolling_mean_by_group(df["es_copia"], 10)
    df["counter_rate_ult10"] = rolling_mean_by_group(df["es_counter"], 10)

    # ------------------------------------------------------------------
    # Feature 7: Entropia de j2 en ultimas 10 rondas
    # ------------------------------------------------------------------
    eps = 1e-9
    p_mat = np.stack(
        [df[f"freq_j2_{jug}_ult10"].to_numpy() for jug in JUGADAS],
        axis=1,
    )
    entropia = -np.sum(p_mat * np.log(p_mat + eps), axis=1)
    df["entropia_j2_ult10"] = entropia

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.

    - Define que columnas usar como features (X)
    - Define la columna target (y) - debe ser 'proxima_jugada_j2'
    - Elimina filas con valores nulos

    Returns:
        (X, y) - Features y target como arrays/DataFrames
    """
    df = df.copy()

    feature_cols = FEATURE_COLS

    # Asegurarse de que no haya NaN en features ni en target
    df = df.dropna(subset=feature_cols + ["proxima_jugada_j2"]).reset_index(drop=True)

    X = df[feature_cols].astype(float)
    y = df["proxima_jugada_j2"].astype(int)

    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    modelos = {
        "KNN_7": KNeighborsClassifier(n_neighbors=7),
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=None),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    mejor_modelo = None
    mejor_nombre = None
    mejor_score = -1.0  # vamos a usar macro-F1

    print("\n==============================")
    print("Evaluacion de modelos")
    print("==============================")

    for nombre, modelo in modelos.items():
        print(f"\n--- Modelo: {nombre} ---")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"Accuracy:  {acc:.3f}")
        print(f"Macro-F1:  {macro_f1:.3f}")
        print("Matriz de confusion:")
        print(confusion_matrix(y_test, y_pred))
        print("\nInforme de clasificacion:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Criterio de seleccion: mejor macro-F1
        if macro_f1 > mejor_score:
            mejor_score = macro_f1
            mejor_modelo = modelo
            mejor_nombre = nombre

    print("\n==============================")
    print(f"Mejor modelo: {mejor_nombre} (Macro-F1 = {mejor_score:.3f})")
    print("==============================")

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

    - Carga un modelo entrenado
    - Mantiene historial de la partida actual
    - Predice la proxima jugada del oponente
    - Decide que jugada hacer para ganar
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []  # Lista de (jugada_j1, jugada_j2) donde j1 = IA, j2 = oponente
        self.ultima_features = None  # Para logica heuristica (copy_rate, etc.)

        if ruta_modelo is None:
            ruta_modelo = RUTA_MODELO

        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print(f"[INFO] Modelo cargado desde: {ruta_modelo}")
        except FileNotFoundError:
            print(
                "[AVISO] Modelo no encontrado. La IA jugara en modo aleatorio "
                "hasta que entrenes y guardes un modelo."
            )

    # ------------------------------------------------------------------
    # Registro de rondas y métricas sobre el historial
    # ------------------------------------------------------------------
    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1 (IA)
            jugada_j2: Jugada del oponente
        """
        self.historial.append((jugada_j1, jugada_j2))

    def _racha_derrotas_reciente(self, ventana: int = 5) -> int:
        """
        Calcula la racha de derrotas consecutivas recientes de la IA (j1).

        Args:
            ventana: numero maximo de rondas hacia atras a considerar

        Returns:
            Numero de derrotas consecutivas mas recientes (0 si no hay)
        """
        racha = 0
        for j1, j2 in reversed(self.historial[-ventana:]):
            res = resultado_ronda(j1, j2)  # j1 = IA
            if res == "j2_gana":  # gana el oponente
                racha += 1
            else:
                break
        return racha

    def _winrate_reciente(self, ventana: int = 10) -> float:
        """
        Calcula el winrate de la IA en las ultimas 'ventana' rondas.

        Args:
            ventana: numero de rondas a considerar

        Returns:
            Proporcion de victorias de la IA (0.0 a 1.0)
        """
        if not self.historial:
            return 0.0

        sub = self.historial[-ventana:]
        ganadas = 0
        total = 0
        for j1, j2 in sub:
            res = resultado_ronda(j1, j2)
            if res == "j1_gana":
                ganadas += 1
            total += 1

        return ganadas / total if total > 0 else 0.0

    def _ia_repitiendo_y_perdiendo(self, k: int = 3) -> bool:
        """
        Devuelve True si en las ultimas k rondas la IA ha jugado SIEMPRE
        la misma mano y ha perdido todas. Sirve para romper bucles tontos
        tipo 'papel pierde 4 veces seguidas contra tijera'.
        """
        if len(self.historial) < k:
            return False

        ult = self.historial[-k:]  # lista de (j1_ia, j2_op)
        jug_ias = [j1 for j1, _ in ult]

        # Todas las jugadas de la IA iguales
        if len(set(jug_ias)) != 1:
            return False

        # Y todas son derrotas de la IA
        for j1, j2 in ult:
            if resultado_ronda(j1, j2) != "j2_gana":
                return False

        return True

    # ------------------------------------------------------------------
    # Prediccion de features (igual que en entrenamiento)
    # ------------------------------------------------------------------
    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.

        - Usa el historial para calcular las mismas features que en entrenamiento
        - Retorna un array con las features en el mismo orden que FEATURE_COLS

        Returns:
            Array con las features para la prediccion (o None si no hay historial)
        """
        if len(self.historial) == 0:
            return None

        # Construimos un DataFrame "falso" de historial como si fuera una partida
        jugadas_j1 = [j1 for j1, _ in self.historial]
        jugadas_j2 = [j2 for _, j2 in self.historial]
        numero_ronda = list(range(1, len(self.historial) + 1))

        df_hist = pd.DataFrame(
            {
                "numero_ronda": numero_ronda,
                "jugada_j1": jugadas_j1,
                "jugada_j2": jugadas_j2,
            }
        )
        # Para reutilizar crear_features con un grupo
        df_hist["partida"] = 1

        # Normalizar texto
        for col in ["jugada_j1", "jugada_j2"]:
            df_hist[col] = df_hist[col].astype(str).str.strip().str.lower()

        # Creamos features usando la MISMA funcion que en entrenamiento
        df_feat = crear_features(df_hist)

        # Tomamos la ultima fila (estado actual)
        ultima_fila = df_feat.iloc[-1]

        # Extraemos las columnas de features en el mismo orden
        features = ultima_fila[FEATURE_COLS].to_numpy(dtype=float)

        # Guardamos ultima fila de features para logica heuristica
        self.ultima_features = ultima_fila

        return features

    # ------------------------------------------------------------------
    # Prediccion con el modelo ML
    # ------------------------------------------------------------------
    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente segun el modelo ML.

        Returns:
            Jugada predicha del oponente (piedra/papel/tijera)
        """
        if self.modelo is None:
            return np.random.choice(list(JUGADA_A_NUM.keys()))

        features = self.obtener_features_actuales()
        if features is None:
            return np.random.choice(list(JUGADA_A_NUM.keys()))

        pred_num = int(self.modelo.predict([features])[0])
        return NUM_A_JUGADA.get(pred_num, "piedra")

    # ------------------------------------------------------------------
    # Prediccion "online" independiente del modelo: frecuencia y ciclos
    # ------------------------------------------------------------------
    def _predecir_por_frecuencia(self, ventana: int = 10) -> str | None:
        """
        Predice la proxima jugada del oponente basandose en la jugada mas frecuente
        en las ultimas 'ventana' rondas.

        Returns:
            Jugada predicha (piedra/papel/tijera) o None si no es posible
        """
        if len(self.historial) < 3:
            return None

        jugadas_op = [j2 for _, j2 in self.historial[-ventana:]]
        counts = {j: jugadas_op.count(j) for j in JUGADAS}
        total = sum(counts.values())
        if total == 0:
            return None

        # Si todas tienen conteo muy parecido, no es muy informativo
        max_j = max(counts, key=counts.get)
        if counts[max_j] < 2:
            return None

        return max_j

    def _predecir_por_patron_ciclico(self, max_longitud: int = 5) -> str | None:
        """
        Intenta detectar un patron ciclico en las jugadas del oponente
        con longitud entre 1 y max_longitud.

        Si detecta que las ultimas 2*L jugadas se repiten (bloque1 == bloque2),
        asume un ciclo de longitud L y predice la siguiente jugada del ciclo.

        Returns:
            Jugada predicha (piedra/papel/tijera) o None si no detecta patron
        """
        jugadas_op = [j2 for _, j2 in self.historial]
        n = len(jugadas_op)
        if n < 4:  # demasiado poco para ver un ciclo
            return None

        for L in range(1, max_longitud + 1):
            if 2 * L > n:
                continue
            bloque1 = jugadas_op[-2 * L : -L]
            bloque2 = jugadas_op[-L:]
            if bloque1 == bloque2:
                # Tenemos algo tipo [A,B,C, A,B,C] → asumimos ciclo [A,B,C]
                idx_siguiente = n % L
                return bloque2[idx_siguiente]

        return None

    # ------------------------------------------------------------------
    # Decision final de jugada (modelo + heuristicas + fallback online)
    # ------------------------------------------------------------------
    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.

        Estrategia:
        1. Usar el modelo ML para predecir la jugada del rival.
        2. Aplicar heuristicas anti-copy, anti-counter, anti-random.
        3. Si el rendimiento reciente es muy malo, activar modo defensa:
           - Intentar detectar ciclos y explotarlos.
           - O usar frecuencia reciente.
           - O, en ultima instancia, aleatorio.
        """
        if self.modelo is None or len(self.historial) == 0:
            return np.random.choice(JUGADAS)

        # Prediccion base con el modelo
        prediccion_oponente = self.predecir_jugada_oponente()
        if prediccion_oponente not in JUGADAS:
            return np.random.choice(JUGADAS)

        # Jugada base: lo que gana a esa prediccion
        decision = PIERDE_CONTRA[prediccion_oponente]

        # Info para heuristicas
        features = getattr(self, "ultima_features", None)
        copy_rate = float(features["copy_rate_ult10"]) if features is not None else 0.0
        counter_rate = (
            float(features["counter_rate_ult10"]) if features is not None else 0.0
        )
        entropia = (
            float(features["entropia_j2_ult10"]) if features is not None else 0.0
        )

        racha_derrotas = self._racha_derrotas_reciente(ventana=5)
        winrate_ult10 = self._winrate_reciente(ventana=10)

        # -----------------------------------------------------------
        # Heuristica 1: el oponente parece copy-bot
        # -----------------------------------------------------------
        if copy_rate > 0.6 and len(self.historial) >= 1:
            ultima_j1 = self.historial[-1][0]
            if ultima_j1 in JUGADAS:
                decision = PIERDE_CONTRA[ultima_j1]

        # -----------------------------------------------------------
        # Heuristica 2: el oponente parece counter-bot
        # -----------------------------------------------------------
        elif counter_rate > 0.6 and len(self.historial) >= 1:
            ultima_j1 = self.historial[-1][0]
            if ultima_j1 in JUGADAS:
                decision = PIERDE_CONTRA[PIERDE_CONTRA[ultima_j1]]

        # -----------------------------------------------------------
        # Heuristica 3: oponente con entropia muy alta (casi random)
        # -> meter algo de ruido para no ser predecible nosotros
        # -----------------------------------------------------------
        if entropia > 1.0:
            if np.random.rand() < 0.3:
                decision = np.random.choice(JUGADAS)

        # -----------------------------------------------------------
        # Heuristica 4: IA repitiendo y perdiendo la misma jugada
        # -----------------------------------------------------------
        if self._ia_repitiendo_y_perdiendo(k=3):
            ult = self.historial[-3:]
            jug_ops = [j2 for _, j2 in ult]
            mas_frecuente_op = max(JUGADAS, key=lambda j: jug_ops.count(j))
            decision = PIERDE_CONTRA[mas_frecuente_op]

        # -----------------------------------------------------------
        # MODO DEFENSA: nos estan reventando
        #   - winrate reciente muy bajo
        #   - o racha de derrotas alta
        #   (ajustado para reaccionar antes)
        # -----------------------------------------------------------
        if len(self.historial) >= 8 and (winrate_ult10 < 0.40 or racha_derrotas >= 3):
            # 1) Intentar detectar un ciclo tipo [1,2,2,3,2] repetido
            jug_ciclo = self._predecir_por_patron_ciclico(max_longitud=5)
            if jug_ciclo in JUGADAS:
                decision = PIERDE_CONTRA[jug_ciclo]
            else:
                # 2) Fallback por frecuencia: jugada mas frecuente reciente
                jug_freq = self._predecir_por_frecuencia(ventana=10)
                if jug_freq in JUGADAS:
                    decision = PIERDE_CONTRA[jug_freq]
                else:
                    # 3) Ultimo recurso: aleatorio puro
                    decision = np.random.choice(JUGADAS)

        return decision

# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """
    Funcion principal para entrenar el modelo.

    Ejecuta: python src/modelo.py
    """
    print("=" * 50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("=" * 50)

    try:
        # 1. Cargar datos
        df = cargar_datos()
        print(f"[INFO] Datos cargados: {len(df)} filas")
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return

    # 2. Preparar datos
    df_prep = preparar_datos(df)
    print(f"[INFO] Datos preparados: {len(df_prep)} filas con target")

    # 3. Crear features
    df_feat = crear_features(df_prep)
    print(f"[INFO] Features creadas. Columnas totales: {len(df_feat.columns)}")

    # 4. Seleccionar features y target
    X, y = seleccionar_features(df_feat)

    # 5. Entrenar modelo
    modelo = entrenar_modelo(X, y)

    # 6. Guardar modelo
    guardar_modelo(modelo, RUTA_MODELO)

    print("\n[OK] Entrenamiento completado y modelo guardado.")


if __name__ == "__main__":
    main()
