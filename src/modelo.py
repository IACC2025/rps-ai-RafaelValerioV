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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"
RUTA_SCALER = RUTA_PROYECTO / "models" / "scaler.pkl"

# Mapeo de jugadas a numeros (para el modelo)
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}

# Constantes auxiliares
JUGADAS = ["piedra", "papel", "tijera"]


# =========================================================
# PARTE 1: EXTRACCION DE DATOS (30% de la nota)
# =========================================================

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

    # Implementacion de la carga de datos
    ruta_csv = Path(ruta_csv)

    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos en: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    # Verificar columnas necesarias
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

    Args:
        df: DataFrame con los datos crudos

    Returns:
        DataFrame preparado para feature engineering
    """
    df = df.copy()

    # Normalizar texto de jugadas
    for col in ["jugada_j1", "jugada_j2"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Ordenar por partida y numero_ronda si existe columna 'partida'
    if "partida" in df.columns:
        df = df.sort_values(["partida", "numero_ronda"]).reset_index(drop=True)
    else:
        df = df.sort_values(["numero_ronda"]).reset_index(drop=True)

    # Convertir jugadas a numeros usando map()
    df["jugada_j1_num"] = df["jugada_j1"].map(JUGADA_A_NUM)
    df["jugada_j2_num"] = df["jugada_j2"].map(JUGADA_A_NUM)

    # Crear target: proxima jugada de j2 usando shift(-1)
    if "partida" in df.columns:
        df["proxima_jugada_j2"] = df.groupby("partida")["jugada_j2_num"].shift(-1)
    else:
        df["proxima_jugada_j2"] = df["jugada_j2_num"].shift(-1)

    # Eliminar filas con valores nulos usando dropna()
    df = df.dropna(
        subset=["jugada_j1_num", "jugada_j2_num", "proxima_jugada_j2"]
    ).reset_index(drop=True)

    # Asegurar tipo entero
    df["jugada_j1_num"] = df["jugada_j1_num"].astype(int)
    df["jugada_j2_num"] = df["jugada_j2_num"].astype(int)
    df["proxima_jugada_j2"] = df["proxima_jugada_j2"].astype(int)

    return df


# =========================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =========================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features (caracteristicas) para el modelo.

    Implementa 3 tipos principales de features:
    1. Frecuencia de cada jugada del oponente (j2)
    2. Ultimas N jugadas (lag features)
    3. Resultado de la ronda anterior

    Features adicionales para mejorar prediccion:
    4. Racha actual (cuantas veces repite la misma jugada)
    5. Patron de comportamiento reactivo (copia jugada anterior)

    NOTA: Se usan ventanas cortas (5 rondas) porque el promedio de
    rondas por partida es 1.5, por lo que ventanas largas (10+) no
    son utiles y causan overfitting.

    Args:
        df: DataFrame con datos preparados

    Returns:
        DataFrame con todas las features creadas
    """
    df = df.copy()

    # Determinar clave de agrupacion
    if "partida" in df.columns and "numero_ronda" in df.columns:
        df = df.sort_values(["partida", "numero_ronda"]).reset_index(drop=True)
        group_key = df["partida"]
    elif "partida" in df.columns:
        df = df.sort_values(["partida"]).reset_index(drop=True)
        group_key = df["partida"]
    else:
        df = df.reset_index(drop=True)
        group_key = None

    # ------------------------------------------
    # Feature 1 - Frecuencia de jugadas
    # ------------------------------------------
    # Calcula que porcentaje de veces j2 juega cada opcion
    # IMPORTANTE: Usa shift(1) para evitar data leakage
    # (no usar info de la ronda actual para predecir la siguiente)

    def rolling_mean_safe(series: pd.Series, window: int) -> pd.Series:
        """Helper para calcular rolling mean por grupo si existe"""
        if group_key is not None:
            return (
                series.groupby(group_key)
                .apply(lambda s: s.shift(1).rolling(window=window, min_periods=0).mean())
                .reset_index(level=0, drop=True).fillna(0)
            )
        else:
            return series.shift(1).rolling(window=window, min_periods=0).mean().fillna(0)

    for jugada in JUGADAS:
        base = (df["jugada_j2"] == jugada).astype(int)
        df[f"freq_j2_{jugada}_ult5"] = rolling_mean_safe(base, 5)

    # ------------------------------------------
    # Feature 2 - Lag features (jugadas anteriores)
    # ------------------------------------------
    # Crea columnas con la ultima jugada de j2 (one-hot encoding)
    # Usa shift(1) para tomar la jugada ANTERIOR, no la actual

    if group_key is not None:
        jugada_j2_anterior = df.groupby(group_key)["jugada_j2"].shift(1)
    else:
        jugada_j2_anterior = df["jugada_j2"].shift(1)

    for jugada in JUGADAS:
        df[f"j2_ult_{jugada}"] = (jugada_j2_anterior == jugada).astype(int)

    # ------------------------------------------
    # Feature 3 - Resultado anterior
    # ------------------------------------------
    # Crea una columna con el resultado de la ronda anterior
    # Esto puede revelar patrones (ej: siempre cambia despues de perder)

    def resultado_ronda(j1: str, j2: str) -> str:
        """Calcula resultado desde punto de vista de j1"""
        if pd.isna(j1) or pd.isna(j2):
            return np.nan
        if j1 == j2:
            return "empate"
        elif GANA_A.get(j1) == j2:
            return "j1_gana"
        else:
            return "j2_gana"

    df["resultado"] = [
        resultado_ronda(j1, j2)
        for j1, j2 in zip(df["jugada_j1"], df["jugada_j2"])
    ]

    if group_key is not None:
        df["resultado_prev"] = df.groupby(group_key)["resultado"].shift(1)
    else:
        df["resultado_prev"] = df["resultado"].shift(1)

    # Feature simple: si j2 gano la ronda anterior
    df["j2_gano_anterior"] = (df["resultado_prev"] == "j2_gana").astype(int)

    # ------------------------------------------
    # Feature 4 - Racha actual
    # ------------------------------------------
    # Cuenta cuantas veces consecutivas j2 repite la misma jugada
    # Usa shift(1) para no incluir la jugada actual

    # Primero crear columna auxiliar con jugada anterior de j2
    if group_key is not None:
        df["jugada_j2_prev_temp"] = df.groupby(group_key)["jugada_j2"].shift(1)
    else:
        df["jugada_j2_prev_temp"] = df["jugada_j2"].shift(1)

    streak_values = []
    if group_key is not None:
        for _, g in df.groupby(group_key, sort=False):
            last = None
            c = 0
            for v in g["jugada_j2_prev_temp"]:
                if pd.isna(v):
                    streak_values.append(0)
                elif v == last:
                    c += 1
                    streak_values.append(c)
                else:
                    c = 1
                    last = v
                    streak_values.append(c)
    else:
        last = None
        c = 0
        for v in df["jugada_j2_prev_temp"]:
            if pd.isna(v):
                streak_values.append(0)
            elif v == last:
                c += 1
                streak_values.append(c)
            else:
                c = 1
                last = v
                streak_values.append(c)

    df["racha_j2_misma_mano"] = streak_values
    df = df.drop("jugada_j2_prev_temp", axis=1)

    # ------------------------------------------
    # Feature 5 - Patron de comportamiento reactivo
    # ------------------------------------------
    # Detecta si j2 copia la jugada anterior de j1 (copy-bot)
    # O si j2 repite su propia jugada anterior

    if group_key is not None:
        df["jugada_j1_prev"] = df.groupby(group_key)["jugada_j1"].shift(1)
        df["jugada_j2_prev"] = df.groupby(group_key)["jugada_j2"].shift(1)
        df["jugada_j2_prev2"] = df.groupby(group_key)["jugada_j2"].shift(2)
    else:
        df["jugada_j1_prev"] = df["jugada_j1"].shift(1)
        df["jugada_j2_prev"] = df["jugada_j2"].shift(1)
        df["jugada_j2_prev2"] = df["jugada_j2"].shift(2)

    # Es copia si jugada_j2_prev == jugada_j1_prev (j2 copió a j1 en la ronda anterior)
    df["es_copia"] = (df["jugada_j2_prev"] == df["jugada_j1_prev"]).astype(int)
    # Es repeticion si jugada_j2_prev == jugada_j2_prev2 (j2 repitió su jugada)
    df["es_repeticion"] = (df["jugada_j2_prev"] == df["jugada_j2_prev2"]).astype(int)

    # Calcular tasas con rolling mean (sin shift adicional, ya está en las variables base)
    if group_key is not None:
        df["copy_rate_ult5"] = (
            df["es_copia"].groupby(group_key)
            .apply(lambda s: s.rolling(window=5, min_periods=0).mean())
            .reset_index(level=0, drop=True).fillna(0)
        )
        df["repite_rate_ult5"] = (
            df["es_repeticion"].groupby(group_key)
            .apply(lambda s: s.rolling(window=5, min_periods=0).mean())
            .reset_index(level=0, drop=True).fillna(0)
        )
    else:
        df["copy_rate_ult5"] = df["es_copia"].rolling(window=5, min_periods=0).mean().fillna(0)
        df["repite_rate_ult5"] = df["es_repeticion"].rolling(window=5, min_periods=0).mean().fillna(0)

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

    # Definir columnas de features
    # Solo features robustas (10 en total) para evitar overfitting
    feature_cols = [
        # Estado inmediato: ultima jugada de j2 (3 features)
        "j2_ult_piedra", "j2_ult_papel", "j2_ult_tijera",
        # Frecuencias en ventana corta (3 features)
        "freq_j2_piedra_ult5", "freq_j2_papel_ult5", "freq_j2_tijera_ult5",
        # Racha actual (1 feature)
        "racha_j2_misma_mano",
        # Comportamiento reactivo (2 features)
        "copy_rate_ult5", "repite_rate_ult5",
        # Contexto de resultado (1 feature)
        "j2_gano_anterior",
    ]

    # Rellenar NaN en features con 0 (primeras rondas sin historial)
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Eliminar solo filas sin target
    df = df.dropna(subset=["proxima_jugada_j2"]).reset_index(drop=True)

    # Crear X (features) e y (target)
    X = df[feature_cols].astype(float)
    y = df["proxima_jugada_j2"].astype(int)

    return X, y


# =========================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =========================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de prediccion.

    - Divide los datos en train/test
    - Entrena al menos 2 modelos diferentes
    - Evalua cada modelo y selecciona el mejor
    - Muestra metricas de evaluacion

    MEJORAS IMPLEMENTADAS:
    - Cross-validation para evaluacion mas robusta
    - Regularizacion fuerte para evitar overfitting
    - Normalizacion de features para Logistic Regression

    Args:
        X: Features
        y: Target (proxima jugada del oponente)
        test_size: Proporcion de datos para test

    Returns:
        Tupla (mejor_modelo, scaler) donde scaler puede ser None
    """
    # Dividir los datos con train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y  # Mantener proporcion de clases
    )

    print(f"\nDatos divididos: {len(X_train)} train, {len(X_test)} test")

    # Normalizar features (importante para Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar varios modelos
    modelos = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            C=10.0,  # Menos regularizacion para mas flexibilidad
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced'  # Balancear clases minoritarias
        ),
        'KNN (k=5)': KNeighborsClassifier(
            n_neighbors=5  # k menor para mas sensibilidad a patrones
        ),
    }

    mejor_modelo = None
    mejor_nombre = None
    mejor_score = -1.0  # Usaremos macro F1 en test para seleccionar
    mejor_scaler = None

    # Cross-validation para evaluacion mas confiable
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluar cada modelo
    for nombre, modelo in modelos.items():
        print(f"\n{'-'*70}")
        print(f" Modelo: {nombre}")
        print(f"{'-'*70}")

        # Determinar si necesita scaling
        X_train_use = X_train_scaled if "Logistic" in nombre else X_train
        X_test_use = X_test_scaled if "Logistic" in nombre else X_test

        # Cross-validation (5-fold)
        cv_scores = cross_val_score(
            modelo, X_train_use, y_train,
            cv=cv, scoring='accuracy'
        )

        print(f"\n Cross-Validation (5-fold):")
        print(f"   - Promedio:  {cv_scores.mean():.3f} (± {cv_scores.std():.3f})")

        # Entrenar en to-do el train set con fit()
        modelo.fit(X_train_use, y_train)

        # Evaluar en train (para detectar overfitting)
        y_train_pred = modelo.predict(X_train_use)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Evaluar en test con predict() y accuracy_score()
        y_test_pred = modelo.predict(X_test_use)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

        print(f"\n Rendimiento:")
        print(f"   - Train accuracy: {train_acc:.3f}")
        print(f"   - Test accuracy:  {test_acc:.3f}")
        print(f"   - Test Macro-F1:  {test_f1_macro:.3f}")
        print(f"   - Diferencia:     {abs(train_acc - test_acc):.3f}")

        # Mostrar classification_report()
        print(f"\n Reporte de clasificacion:")
        print(classification_report(
            y_test, y_test_pred,
            target_names=["Piedra", "Papel", "Tijera"],
            zero_division=0
        ))

        # Mostrar matriz de confusion
        print(f" Matriz de confusion:")
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"              Pred:  Piedra  Papel  Tijera")
        for i, real in enumerate(["Piedra", "Papel", "Tijera"]):
            print(f"   Real {real:6s}:     {cm[i][0]:2d}     {cm[i][1]:2d}     {cm[i][2]:2d}")

        # Seleccionar mejor modelo basado en Macro F1 (mejor balance de clases)
        if test_f1_macro > mejor_score:
            mejor_score = test_f1_macro
            mejor_modelo = modelo
            mejor_nombre = nombre
            mejor_scaler = scaler if "Logistic" in nombre else None

    print("\n" + "="*70)
    print(f" MEJOR MODELO: {mejor_nombre}")
    print(f"   - Test Macro-F1: {mejor_score:.3f}")
    print("="*70)

    # Retornar mejor modelo y scaler
    return mejor_modelo, mejor_scaler


def guardar_modelo(modelo, scaler=None, ruta: str = None):
    """Guarda el modelo entrenado en un archivo."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")

    # Guardar scaler si existe
    if scaler is not None:
        ruta_scaler = RUTA_SCALER
        with open(ruta_scaler, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Scaler guardado en: {ruta_scaler}")


def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")

    with open(ruta, "rb") as f:
        modelo = pickle.load(f)

    # Intentar cargar scaler
    scaler = None
    if os.path.exists(RUTA_SCALER):
        with open(RUTA_SCALER, "rb") as f:
            scaler = pickle.load(f)

    return modelo, scaler


# =========================================================
# PARTE 4: PREDICCION Y JUEGO
# =========================================================

class JugadorIA:
    """
    Clase que encapsula el modelo para jugar.

    Funcionalidades:
    - Cargar un modelo entrenado
    - Mantener historial de la partida actual
    - Predecir la proxima jugada del oponente
    - Decidir que jugada hacer para ganar

    MEJORAS IMPLEMENTADAS:
    - Detector de oponente aleatorio
    - Detector de sesgo fuerte
    - Sistema de decision adaptativo
    - Modo defensa conservador
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.scaler = None
        self.historial = []  # Lista de (jugada_j1, jugada_j2)

        # Cargar el modelo si existe
        try:
            self.modelo, self.scaler = cargar_modelo(ruta_modelo)
        except FileNotFoundError:
            print("Modelo no encontrado. Entrena primero con main()")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1 (IA)
            jugada_j2: Jugada del oponente
        """
        self.historial.append((jugada_j1, jugada_j2))

    def _es_oponente_aleatorio(self, ventana: int = 15) -> bool:
        """
        Detecta si el oponente parece jugar aleatoriamente.

        Criterio: todas las jugadas tienen frecuencia entre 25-42%
        """
        if len(self.historial) < ventana:
            return False

        jugadas_op = [j2 for _, j2 in self.historial[-ventana:]]
        counts = {j: jugadas_op.count(j) for j in JUGADAS}

        freqs = [counts[j] / ventana for j in JUGADAS]
        max_freq = max(freqs)
        min_freq = min(freqs)

        # Si todas estan equilibradas → aleatorio
        return max_freq < 0.42 and min_freq > 0.25

    def _tiene_sesgo_fuerte(self, ventana: int = 15) -> tuple:
        """
        Detecta si hay un sesgo fuerte hacia una jugada.

        Returns:
            (tiene_sesgo, jugada_dominante)
        """
        if len(self.historial) < 8:
            return False, None

        jugadas_op = [j2 for _, j2 in self.historial[-ventana:]]
        counts = {j: jugadas_op.count(j) for j in JUGADAS}

        max_jugada = max(counts, key=counts.get)
        max_freq = counts[max_jugada] / len(jugadas_op)

        # Sesgo fuerte si una jugada aparece >40%
        return max_freq > 0.40, max_jugada

    def _detectar_counter_bot(self, ventana: int = 10) -> bool:
        """
        Detecta si el oponente esta jugando como counter-bot.

        Counter-bot: juega lo que gana a tu jugada anterior.

        Returns:
            True si detecta comportamiento de counter-bot
        """
        if len(self.historial) < ventana:
            return False

        # Contar cuantas veces el oponente juega lo que gana a mi jugada anterior
        counter_hits = 0
        total = 0

        for i in range(len(self.historial) - ventana, len(self.historial)):
            if i == 0:
                continue  # No hay jugada anterior mia

            mi_jugada_anterior = self.historial[i-1][0]  # j1 de ronda anterior
            jugada_oponente_actual = self.historial[i][1]  # j2 de ronda actual

            # Si oponente jugo lo que gana a mi jugada anterior
            if PIERDE_CONTRA[mi_jugada_anterior] == jugada_oponente_actual:
                counter_hits += 1
            total += 1

        if total == 0:
            return False

        counter_rate = counter_hits / total

        # Si >60% del tiempo hace counter → es counter-bot
        return counter_rate > 0.60

    def _detectar_patron_ciclico(self, ventana: int = 15) -> tuple:
        """
        Detecta si el oponente sigue un patron ciclico simple (ciclo de 2, 3, 4 o 5).

        Returns:
            (es_ciclico, ciclo) donde ciclo es la lista de jugadas que se repiten
        """
        if len(self.historial) < 8:
            return False, None

        jugadas_op = [j2 for _, j2 in self.historial[-ventana:]]
        n = len(jugadas_op)

        # Probar diferentes longitudes de ciclo
        for ciclo_len in [2, 3, 4, 5]:
            if n < ciclo_len * 2:  # Necesitamos al menos 2 repeticiones
                continue

            # Tomar el patrón propuesto de las primeras N jugadas
            patron = jugadas_op[:ciclo_len]

            # Contar cuántas veces se repite
            matches = 0
            total_checks = 0

            for i in range(ciclo_len, n):
                expected = patron[i % ciclo_len]
                if jugadas_op[i] == expected:
                    matches += 1
                total_checks += 1

            if total_checks == 0:
                continue

            # Si >70% coincide, es un ciclo
            match_rate = matches / total_checks
            if match_rate > 0.70:
                return True, patron

        return False, None

    def _baseline_estadistico(self, ventana: int = 10) -> str:
        """Predice la jugada mas frecuente reciente del oponente"""
        if len(self.historial) < 3:
            return "piedra"  # Default conocido del dataset

        jugadas_op = [j2 for _, j2 in self.historial[-ventana:]]
        counts = {j: jugadas_op.count(j) for j in JUGADAS}
        return max(counts, key=counts.get)

    def _winrate_reciente(self, ventana: int = 10) -> float:
        """Calcula winrate de la IA en ultimas N rondas"""
        if not self.historial:
            return 0.5

        sub = self.historial[-ventana:]
        ganadas = 0
        for j1, j2 in sub:
            if j1 == j2:
                continue
            elif GANA_A.get(j1) == j2:
                ganadas += 1

        return ganadas / len(sub) if sub else 0.5

    def _racha_derrotas(self, ventana: int = 10) -> int:
        """Cuenta derrotas consecutivas recientes"""
        racha = 0
        for j1, j2 in reversed(self.historial[-ventana:]):
            # Perdemos si j2 != j1 y j2 no es lo que gana j1
            if j1 != j2 and GANA_A.get(j1) != j2:
                racha += 1
            else:
                break
        return racha

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.

        - Usa el historial para calcular las mismas features que usaste en entrenamiento
        - Retorna un array con las features

        Returns:
            Array con las features para la prediccion (o None si no hay historial)
        """
        if len(self.historial) == 0:
            return None

        # Construir DataFrame temporal del historial
        jugadas_j1 = [j1 for j1, _ in self.historial]
        jugadas_j2 = [j2 for _, j2 in self.historial]
        numero_ronda = list(range(1, len(self.historial) + 1))

        df_hist = pd.DataFrame({
            "numero_ronda": numero_ronda,
            "jugada_j1": jugadas_j1,
            "jugada_j2": jugadas_j2,
            "partida": 1  # Una sola partida
        })

        # Normalizar texto
        for col in ["jugada_j1", "jugada_j2"]:
            df_hist[col] = df_hist[col].astype(str).str.strip().str.lower()

        # Crear features usando la MISMA funcion que en entrenamiento
        df_feat = crear_features(df_hist)

        # Extraer ultima fila (estado actual)
        ultima_fila = df_feat.iloc[-1]

        # Features en el mismo orden que en seleccionar_features
        feature_cols = [
            "j2_ult_piedra", "j2_ult_papel", "j2_ult_tijera",
            "freq_j2_piedra_ult5", "freq_j2_papel_ult5", "freq_j2_tijera_ult5",
            "racha_j2_misma_mano",
            "copy_rate_ult5", "repite_rate_ult5",
            "j2_gano_anterior",
        ]

        features = ultima_fila[feature_cols].to_numpy(dtype=float)

        return features

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.

        - Usa obtener_features_actuales() para obtener las features
        - Usa el modelo para predecir
        - Convierte la prediccion numerica a texto

        Returns:
            Jugada predicha del oponente (piedra/papel/tijera)
        """
        if self.modelo is None:
            # Si no hay modelo, juega aleatorio
            return np.random.choice(["piedra", "papel", "tijera"])

        # Obtener features del historial actual
        features = self.obtener_features_actuales()
        if features is None:
            return self._baseline_estadistico()

        # Aplicar scaling si es necesario
        if self.scaler is not None:
            features = self.scaler.transform([features])[0]

        try:
            # Usar el modelo para predecir
            prediccion = self.modelo.predict([features])[0]
            # Convertir numero a texto
            return NUM_A_JUGADA[prediccion]
        except:
            return self._baseline_estadistico()

    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.

        Sistema de decision adaptativo (en orden de prioridad):
        1. Primeras rondas: usar modelo ML
        2. Anti-patron predecible: añadir aleatoriedad tras victorias/empates
        3. Si oponente es counter-bot: estrategia anti-counter
        4. Si oponente sigue patron ciclico: predecir y explotar ciclo
        5. Si oponente aleatorio: jugar aleatorio tambien
        6. Si tiene sesgo fuerte: explotarlo
        7. Si winrate muy bajo: modo defensa
        8. Sino: usar modelo ML

        Returns:
            La jugada que gana a la prediccion del oponente
        """
        n_rondas = len(self.historial)

        # Primeras rondas: confiar en modelo
        if n_rondas < 5:
            prediccion_oponente = self.predecir_jugada_oponente()
            if prediccion_oponente is None:
                return np.random.choice(["piedra", "papel", "tijera"])
            # Juega lo que le gana a la prediccion
            return PIERDE_CONTRA[prediccion_oponente]

        # ANTI-PATRON: No ser predecible tras victorias o empates
        # Problema detectado: humanos aprenden que repetimos tras ganar/empatar
        if n_rondas >= 2:
            ultima_ronda = self.historial[-1]
            mi_jugada_ant = ultima_ronda[0]
            jugada_op_ant = ultima_ronda[1]

            # ¿Ganamos o empatamos la ronda anterior?
            gane_anterior = (GANA_A.get(mi_jugada_ant) == jugada_op_ant)
            empate_anterior = (mi_jugada_ant == jugada_op_ant)

            # Si ganamos o empatamos: 45% probabilidad de jugar aleatorio
            # Esto rompe el patron de "repetir tras ganar/empatar"
            if (gane_anterior or empate_anterior) and np.random.rand() < 0.45:
                return np.random.choice(JUGADAS)

        # DETECTOR PRIORITARIO: Counter-bot (debe ir antes que otros)
        if n_rondas >= 10 and self._detectar_counter_bot(ventana=10):
            # Estrategia anti-counter CORRECTA:
            # Counter-bot juega lo que gana a mi jugada ANTERIOR
            # Ejemplo:
            #   - Ronda N: Yo jugué "piedra"
            #   - Ronda N+1: Counter jugará "papel" (gana a piedra)
            #   - Ronda N+1: Yo debo jugar "tijera" (gana a papel)
            #
            # Formula: jugar GANA_A[PIERDE_CONTRA[mi_jugada_anterior]]

            if len(self.historial) > 0:
                mi_jugada_anterior = self.historial[-1][0]
                # Lo que counter jugará (lo que gana a mi jugada anterior)
                prediccion_counter = PIERDE_CONTRA[mi_jugada_anterior]
                # Lo que yo juego para ganarle
                return PIERDE_CONTRA[prediccion_counter]
            else:
                # Primera ronda, jugar aleatorio
                return np.random.choice(JUGADAS)

        # DETECTOR 2: Patron ciclico (alternancia o ciclo de 3-5)
        if n_rondas >= 8:
            es_ciclico, ciclo = self._detectar_patron_ciclico(ventana=min(15, n_rondas))
            if es_ciclico and ciclo:
                # Predecir siguiente jugada del ciclo
                posicion_actual = (len(self.historial)) % len(ciclo)
                prediccion_ciclo = ciclo[posicion_actual]
                # 25% del tiempo: aleatorizar para no ser predecible
                if np.random.rand() < 0.25:
                    return np.random.choice(JUGADAS)
                # Jugar lo que gana a esa prediccion
                return PIERDE_CONTRA[prediccion_ciclo]

        # DETECTOR 3: Oponente aleatorio
        if n_rondas >= 15 and self._es_oponente_aleatorio(ventana=15):
            # Si oponente es aleatorio, jugar aleatorio tambien
            decision = np.random.choice(JUGADAS)
            # Pequeno sesgo hacia papel (por sesgo conocido del dataset)
            if np.random.rand() < 0.15:
                decision = "papel"
            return decision

        # DETECTOR 4: Sesgo fuerte
        tiene_sesgo, jugada_dominante = self._tiene_sesgo_fuerte(ventana=min(20, n_rondas))
        if tiene_sesgo and jugada_dominante:
            # Explotar el sesgo directamente
            # 15% del tiempo: aleatorizar para evitar counter-exploitation
            if np.random.rand() < 0.15:
                return np.random.choice(JUGADAS)
            return PIERDE_CONTRA[jugada_dominante]

        # DETECTOR 5: Modo defensa (solo si vamos MUY mal)
        winrate = self._winrate_reciente(ventana=min(15, n_rondas))
        racha = self._racha_derrotas(ventana=8)

        if n_rondas >= 15 and (winrate < 0.25 or racha >= 6):
            # Volver a baseline estadistico
            prediccion_baseline = self._baseline_estadistico(ventana=10)
            return PIERDE_CONTRA[prediccion_baseline]

        # MODO NORMAL: Usar modelo ML
        prediccion_oponente = self.predecir_jugada_oponente()

        if prediccion_oponente is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        # CRITICO: Añadir aleatoriedad incluso cuando usamos el modelo
        # Un humano inteligente puede aprender los patrones del modelo ML
        # 18% del tiempo: ignorar prediccion y jugar aleatorio
        if np.random.rand() < 0.18:
            return np.random.choice(JUGADAS)

        # Juega lo que le gana a la prediccion
        return PIERDE_CONTRA[prediccion_oponente]


# =========================================================
# FUNCION PRINCIPAL
# =========================================================

def main():
    """
    Funcion principal para entrenar el modelo.

    Ejecuta: python src/modelo.py
    """
    print("="*50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*50)

    # Implementacion del flujo completo:

    # 1. Cargar datos
    try:
        df = cargar_datos()
        print(f"Datos cargados: {len(df)} filas")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return

    # 2. Preparar datos
    df_prep = preparar_datos(df)
    print(f"Datos preparados: {len(df_prep)} filas con target")

    # 3. Crear features
    df_feat = crear_features(df_prep)
    print(f"Features creadas")

    # 4. Seleccionar features
    X, y = seleccionar_features(df_feat)

    # 5. Entrenar modelo
    modelo, scaler = entrenar_modelo(X, y)

    # 6. Guardar modelo
    guardar_modelo(modelo, scaler)

    print("\n" + "="*50)
    print("Entrenamiento completado")
    print("="*50)


if __name__ == "__main__":
    main()