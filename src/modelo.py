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

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


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

    Args:
        ruta_csv: Ruta al archivo CSV (usa RUTA_DATOS por defecto)

    Returns:
        DataFrame con los datos de las partidas
    """
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encuentra el archivo: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    # Limpieza de espacios en blanco
    df['jugada_j1'] = df['jugada_j1'].str.strip()
    df['jugada_j2'] = df['jugada_j2'].str.strip()

    columnas_requeridas = ['numero_ronda', 'jugada_j1', 'jugada_j2']
    for col in columnas_requeridas:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    print(f"Datos cargados: {len(df)} rondas")
    print(f"Distribución jugada_j2: \n{df['jugada_j2'].value_counts()}\n")

    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelo.

    Args:
        df: DataFrame con los datos crudos

    Returns:
        DataFrame preparado para feature engineering
    """
    df = df.copy()

    # Convertir jugadas a numeros
    df['j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)

    # Crear la columna objetivo: proxima jugada del oponente
    df['proxima_jugada_j2'] = df['j2_num'].shift(-1)

    # Calcular resultado de cada ronda
    df['resultado'] = df.apply(lambda row: calcular_resultado(row['j1_num'], row['j2_num']), axis=1)

    # Eliminar ultima fila (no tiene proxima jugada)
    df = df[:-1].copy()
    df = df.dropna()

    return df


def calcular_resultado(j1, j2):
    """Calcula el resultado: 1=gana j1, 0=empate, -1=gana j2"""
    if j1 == j2:
        return 0
    if (j1 == 0 and j2 == 2) or (j1 == 1 and j2 == 0) or (j1 == 2 and j2 == 1):
        return 1
    return -1


# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features (caracteristicas) para el modelo.

    Args:
        df: DataFrame con datos preparados

    Returns:
        DataFrame con todas las features creadas
    """
    df = df.copy()

    # Frecuencias historicas del oponente
    df['freq_piedra_j2'] = (df['j2_num'] == 0).expanding().mean()
    df['freq_papel_j2'] = (df['j2_num'] == 1).expanding().mean()
    df['freq_tijera_j2'] = (df['j2_num'] == 2).expanding().mean()

    # Ultimas jugadas del oponente
    df['j2_lag1'] = df['j2_num'].shift(1)
    df['j2_lag2'] = df['j2_num'].shift(2)
    df['j2_lag3'] = df['j2_num'].shift(3)

    # Ultimas jugadas propias
    df['j1_lag1'] = df['j1_num'].shift(1)
    df['j1_lag2'] = df['j1_num'].shift(2)

    # Resultado de rondas anteriores
    df['resultado_lag1'] = df['resultado'].shift(1)

    # Patron: repite la misma jugada
    df['j2_repite'] = (df['j2_num'] == df['j2_lag1']).astype(int)

    # Patron: ciclo (piedra->papel->tijera)
    df['j2_ciclo'] = ((df['j2_lag1'] - df['j2_num']) % 3 == 2).astype(int)

    # Frecuencia en ventana movil de 10
    df['freq_piedra_j2_w10'] = (df['j2_num'] == 0).rolling(window=10, min_periods=1).mean()
    df['freq_papel_j2_w10'] = (df['j2_num'] == 1).rolling(window=10, min_periods=1).mean()
    df['freq_tijera_j2_w10'] = (df['j2_num'] == 2).rolling(window=10, min_periods=1).mean()

    # Cambios de jugada
    df['j2_cambio'] = (df['j2_num'] != df['j2_lag1']).astype(int)

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.

    Returns:
        (X, y) - Features y target como arrays/DataFrames
    """
    feature_cols = [
        'freq_piedra_j2', 'freq_papel_j2', 'freq_tijera_j2',
        'j2_lag1', 'j2_lag2', 'j2_lag3',
        'j1_lag1', 'j1_lag2',
        'resultado_lag1',
        'j2_repite', 'j2_ciclo',
        'freq_piedra_j2_w10', 'freq_papel_j2_w10', 'freq_tijera_j2_w10',
        'j2_cambio'
    ]

    # Eliminar filas con NaN en features o target
    df_clean = df.dropna(subset=feature_cols + ['proxima_jugada_j2'])

    X = df_clean[feature_cols]
    y = df_clean['proxima_jugada_j2']

    print(f"Features seleccionadas: {len(feature_cols)}")
    print(f"Datos validos: {len(X)}")

    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de prediccion.

    Args:
        X: Features
        y: Target (proxima jugada del oponente)
        test_size: Proporcion de datos para test

    Returns:
        El mejor modelo entrenado
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )

    modelos = {
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=6,
                                               min_samples_split=10, class_weight='balanced',
                                               random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                                       random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'DecisionTree': DecisionTreeClassifier(max_depth=5, min_samples_split=10,
                                               class_weight='balanced', random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced',
                                                 C=0.1, random_state=42)
    }

    resultados = {}
    mejor_score = -1
    mejor_modelo = None
    mejor_nombre = None

    print("\n" + "="*60)
    print("EVALUACION DE MODELOS")
    print("="*60)

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)

        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)

        print(f"\n{nombre}:")
        print(f"  Accuracy Train: {acc_train:.4f}")
        print(f"  Accuracy Test:  {acc_test:.4f}")
        print(f"  Diferencia:     {abs(acc_train - acc_test):.4f}")

        overfitting = acc_train - acc_test
        if overfitting > 0.20:
            print(f"  Posible overfitting detectado")

        report = classification_report(y_test, y_pred_test,
                                      target_names=['piedra', 'papel', 'tijera'],
                                      zero_division=0, output_dict=True)

        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_pred_test,
                                   target_names=['piedra', 'papel', 'tijera'],
                                   zero_division=0))

        print("Matriz de Confusion:")
        cm = confusion_matrix(y_test, y_pred_test)
        print(cm)

        # Verificar que no haya clases con precision 0
        min_precision = min(report['piedra']['precision'],
                          report['papel']['precision'],
                          report['tijera']['precision'])

        # Score combinado: accuracy test - penalización por overfitting - penalización por clases débiles
        score = acc_test - (overfitting * 0.5) - (0 if min_precision > 0.15 else 0.2)

        resultados[nombre] = {
            'modelo': modelo,
            'acc_test': acc_test,
            'overfitting': overfitting,
            'score': score,
            'min_precision': min_precision
        }

        if score > mejor_score:
            mejor_score = score
            mejor_modelo = modelo
            mejor_nombre = nombre

    print("\n" + "="*60)
    print(f"MEJOR MODELO: {mejor_nombre}")
    print(f"Test Accuracy: {resultados[mejor_nombre]['acc_test']:.4f}")
    print(f"Score: {mejor_score:.4f}")
    print("="*60 + "\n")

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
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []

        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("Modelo cargado correctamente")
        except FileNotFoundError:
            print("⚠ Modelo no encontrado. Entrena primero ejecutando main()")

        # Variables para detección de patrones
        self.patron_detectado = None
        self.patron_confianza = 0

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1
            jugada_j2: Jugada del oponente
        """
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.

        Returns:
            Array con las features para la prediccion
        """
        if len(self.historial) < 3:
            return np.array([0.33, 0.33, 0.33] + [0]*12)

        j1_hist = np.array([JUGADA_A_NUM[j[0]] for j in self.historial])
        j2_hist = np.array([JUGADA_A_NUM[j[1]] for j in self.historial])

        features = []

        # Frecuencias historicas
        features.append(np.mean(j2_hist == 0))
        features.append(np.mean(j2_hist == 1))
        features.append(np.mean(j2_hist == 2))

        # Lags
        features.append(j2_hist[-1] if len(j2_hist) >= 1 else 0)
        features.append(j2_hist[-2] if len(j2_hist) >= 2 else 0)
        features.append(j2_hist[-3] if len(j2_hist) >= 3 else 0)
        features.append(j1_hist[-1] if len(j1_hist) >= 1 else 0)
        features.append(j1_hist[-2] if len(j1_hist) >= 2 else 0)

        # Resultado anterior
        if len(self.historial) >= 1:
            features.append(calcular_resultado(j1_hist[-1], j2_hist[-1]))
        else:
            features.append(0)

        # Patrones
        features.append(int(len(j2_hist) >= 2 and j2_hist[-1] == j2_hist[-2]))
        features.append(int(len(j2_hist) >= 2 and (j2_hist[-2] - j2_hist[-1]) % 3 == 2))

        # Frecuencias ventana movil 10
        recent = j2_hist[-10:] if len(j2_hist) >= 10 else j2_hist
        features.append(np.mean(recent == 0))
        features.append(np.mean(recent == 1))
        features.append(np.mean(recent == 2))

        # Cambio
        features.append(int(len(j2_hist) >= 2 and j2_hist[-1] != j2_hist[-2]))

        return np.array(features).reshape(1, -1)

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.

        Returns:
            Jugada predicha del oponente (piedra/papel/tijera)
        """
        if self.modelo is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        if len(self.historial) < 3:
            # Al inicio, basarse en estadisticas generales de Victor
            return "piedra"  # Victor tiene sesgo hacia piedra

        features = self.obtener_features_actuales()
        prediccion = self.modelo.predict(features)[0]

        return NUM_A_JUGADA[int(prediccion)]

    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.
        Sistema con detección agresiva de patrones.

        Returns:
            La jugada que gana a la prediccion del oponente
        """
        # Si hay poco historial, explotar sesgo de Victor hacia piedra
        if len(self.historial) < 5:
            return "papel"

        # Detectar y bloquear en patrones específicos
        patron_activo = self._detectar_y_bloquear_patron()
        if patron_activo:
            return patron_activo

        # Si no hay patrón claro, usar modelo ML
        prediccion_oponente = self.predecir_jugada_oponente()
        return PIERDE_CONTRA[prediccion_oponente]

    def _detectar_y_bloquear_patron(self) -> str:
        """
        Detecta patrones del oponente y retorna la contramedida.
        Se bloquea en el patrón detectado hasta que deje de funcionar.
        """
        if len(self.historial) < 6:
            return None

        # Verificar si el patrón actual sigue funcionando
        if self.patron_detectado:
            if self._verificar_patron_activo():
                # Patrón sigue funcionando, continuar con contramedida
                return self._contramedida_patron(self.patron_detectado)
            else:
                # Patrón dejó de funcionar, resetear
                self.patron_detectado = None
                self.patron_confianza = 0

        # Buscar nuevos patrones (orden de prioridad)
        # Primero los patrones fáciles y confiables
        patrones = [
            ('constante', self._detectar_constante),
            ('ciclo', self._detectar_ciclo),
            ('copy', self._detectar_copy_bot),
            ('sesgo_fuerte', self._detectar_sesgo_fuerte),
            ('counter', self._detectar_counter_bot),
        ]

        for nombre, detector in patrones:
            if detector():
                self.patron_detectado = nombre
                self.patron_confianza = len(self.historial)
                return self._contramedida_patron(nombre)

        # Solo buscar patrones difíciles si tenemos suficiente historial
        if len(self.historial) >= 15:
            patrones_dificiles = [
                ('cambia_perder', self._detectar_cambia_tras_perder),
                ('anti_counter', self._detectar_anti_counter),
            ]

            for nombre, detector in patrones_dificiles:
                if detector():
                    self.patron_detectado = nombre
                    self.patron_confianza = len(self.historial)
                    return self._contramedida_patron(nombre)

        return None

    def _verificar_patron_activo(self) -> bool:
        """Verifica si el patrón detectado sigue activo (ganamos)"""
        if len(self.historial) < self.patron_confianza + 4:
            return True

        # Ver últimas 4 rondas desde que detectamos el patrón
        ultimas = self.historial[-(len(self.historial) - self.patron_confianza):]
        if len(ultimas) < 4:
            return True

        ultimas_4 = ultimas[-4:]
        victorias = sum(1 for j1, j2 in ultimas_4
                       if calcular_resultado(JUGADA_A_NUM[j1], JUGADA_A_NUM[j2]) == 1)
        derrotas = sum(1 for j1, j2 in ultimas_4
                      if calcular_resultado(JUGADA_A_NUM[j1], JUGADA_A_NUM[j2]) == -1)

        # Si ganamos al menos 2 de 4, el patrón sigue activo
        # O si no estamos perdiendo mayoría
        return victorias >= 2 or derrotas < 3

    def _detectar_constante(self) -> bool:
        """Detecta si juega siempre lo mismo"""
        ultimas_8 = [j2 for _, j2 in self.historial[-8:]]
        return len(set(ultimas_8)) == 1

    def _detectar_ciclo(self) -> bool:
        """Detecta ciclo perfecto piedra->papel->tijera"""
        ultimas_6 = [j2 for _, j2 in self.historial[-6:]]
        ciclo = ["piedra", "papel", "tijera"]

        for i in range(len(ultimas_6)-1):
            idx_actual = ciclo.index(ultimas_6[i])
            idx_siguiente = ciclo.index(ultimas_6[i+1])
            if (idx_actual + 1) % 3 != idx_siguiente:
                return False
        return True

    def _detectar_counter_bot(self) -> bool:
        """Detecta si juega lo que nos gana"""
        if len(self.historial) < 6:
            return False

        aciertos = 0
        for i in range(-6, -1):
            nuestra = self.historial[i][0]
            siguiente_oponente = self.historial[i+1][1]
            if PIERDE_CONTRA[nuestra] == siguiente_oponente:
                aciertos += 1

        return aciertos >= 5

    def _detectar_copy_bot(self) -> bool:
        """Detecta si copia nuestra jugada"""
        if len(self.historial) < 6:
            return False

        aciertos = 0
        for i in range(-6, -1):
            nuestra = self.historial[i][0]
            siguiente_oponente = self.historial[i+1][1]
            if nuestra == siguiente_oponente:
                aciertos += 1

        return aciertos >= 5

    def _detectar_sesgo_fuerte(self) -> bool:
        """Detecta sesgo muy fuerte hacia una opción (>70%)"""
        ultimas_10 = [j2 for _, j2 in self.historial[-10:]]
        from collections import Counter
        contador = Counter(ultimas_10)
        mas_comun = contador.most_common(1)[0]
        return mas_comun[1] >= 7

    def _detectar_cambia_tras_perder(self) -> bool:
        """Detecta si siempre cambia después de perder"""
        if len(self.historial) < 6:
            return False

        # Analizar últimas 6 rondas
        cambios_tras_perder = 0
        ocasiones_perdio = 0
        mantiene_tras_ganar = 0
        ocasiones_gano = 0

        for i in range(len(self.historial) - 6, len(self.historial) - 1):
            j1, j2 = self.historial[i]
            resultado = calcular_resultado(JUGADA_A_NUM[j1], JUGADA_A_NUM[j2])
            j2_siguiente = self.historial[i+1][1]

            if resultado == 1:  # Oponente perdió
                ocasiones_perdio += 1
                if j2 != j2_siguiente:
                    cambios_tras_perder += 1
            elif resultado == -1:  # Oponente ganó
                ocasiones_gano += 1
                if j2 == j2_siguiente:
                    mantiene_tras_ganar += 1

        # Patrón: cambia cuando pierde (>60%) Y tiende a mantener cuando gana
        if ocasiones_perdio >= 2:
            ratio_cambia = cambios_tras_perder / ocasiones_perdio
            if ratio_cambia >= 0.6:
                return True

        return False

    def _detectar_anti_counter(self) -> bool:
        """Detecta anti-counter-bot: juega lo que gana a lo que nos ganaria"""
        if len(self.historial) < 6:
            return False

        aciertos = 0
        for i in range(-6, -1):
            # Lo que le ganaria al oponente en ronda i
            suya = self.historial[i][1]
            counter_a_suya = PIERDE_CONTRA[suya]
            # Lo que le gana a ese counter
            anti_counter = PIERDE_CONTRA[counter_a_suya]

            # Ver si en i+1 juega eso
            siguiente = self.historial[i+1][1]
            if siguiente == anti_counter:
                aciertos += 1

        return aciertos >= 4

    def _contramedida_patron(self, patron: str) -> str:
        """Retorna la jugada óptima contra el patrón detectado"""
        if patron == 'constante':
            # Jugar lo que le gana
            ultima = self.historial[-1][1]
            return PIERDE_CONTRA[ultima]

        elif patron == 'ciclo':
            # Predecir siguiente en ciclo y jugar lo que le gana
            ciclo = ["piedra", "papel", "tijera"]
            ultima = self.historial[-1][1]
            idx = ciclo.index(ultima)
            proxima = ciclo[(idx + 1) % 3]
            return PIERDE_CONTRA[proxima]

        elif patron == 'counter':
            # Jugar aleatorio para empatar teóricamente
            return np.random.choice(["piedra", "papel", "tijera"])

        elif patron == 'copy':
            # Jugar lo que nos ganaría a nosotros mismos
            nuestra_ultima = self.historial[-1][0]
            return PIERDE_CONTRA[nuestra_ultima]

        elif patron == 'sesgo_fuerte':
            # Jugar lo que le gana a su opción favorita
            ultimas_10 = [j2 for _, j2 in self.historial[-10:]]
            from collections import Counter
            mas_comun = Counter(ultimas_10).most_common(1)[0][0]
            return PIERDE_CONTRA[mas_comun]

        elif patron == 'cambia_perder':
            # Estrategia mixta:
            # 1. Si acabamos de hacer perder al oponente, predecir que cambiará
            # 2. Jugar lo que le gana a lo que MENOS ha jugado recientemente
            ultima_oponente = self.historial[-1][1]
            resultado_ultimo = calcular_resultado(
                JUGADA_A_NUM[self.historial[-1][0]],
                JUGADA_A_NUM[ultima_oponente]
            )

            ultimas_6 = [j2 for _, j2 in self.historial[-6:]]
            from collections import Counter
            contador = Counter(ultimas_6)

            if resultado_ultimo == 1:  # Oponente perdió, cambiará
                # Opciones a las que puede cambiar (las otras dos)
                opciones = ["piedra", "papel", "tijera"]
                opciones.remove(ultima_oponente)

                # De esas dos, predecir la que menos ha jugado recientemente
                # (tiende a rotar)
                menos_jugada = min(opciones, key=lambda x: contador.get(x, 0))
                return PIERDE_CONTRA[menos_jugada]
            else:
                # No perdió, usar frecuencias recientes
                mas_comun = contador.most_common(1)[0][0]
                return PIERDE_CONTRA[mas_comun]

        elif patron == 'anti_counter':
            # Para anti-counter: jugar la menos jugada por nosotros
            # Esto rompe el meta-juego
            ultimas_8_nuestras = [j1 for j1, _ in self.historial[-8:]]
            from collections import Counter
            contador = Counter(ultimas_8_nuestras)
            menos_usada = min(["piedra", "papel", "tijera"],
                            key=lambda x: contador.get(x, 0))
            return menos_usada

        return np.random.choice(["piedra", "papel", "tijera"])


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """
    Funcion principal para entrenar el modelo.

    Ejecuta: python src/modelo.py
    """
    print("="*60)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*60)

    # Cargar y preparar datos
    print("\n1. Cargando datos...")
    df = cargar_datos()

    print("\n2. Preparando datos...")
    df = preparar_datos(df)

    print("\n3. Creando features...")
    df = crear_features(df)

    print("\n4. Seleccionando features...")
    X, y = seleccionar_features(df)

    print("\n5. Entrenando modelos...")
    mejor_modelo = entrenar_modelo(X, y)

    print("\n6. Guardando modelo...")
    guardar_modelo(mejor_modelo)

    print("\n✓ Proceso completado exitosamente")
    print("\nAhora puedes usar JugadorIA para jugar contra el modelo")


if __name__ == "__main__":
    main()