# Documentacion de Recogida de Datos

**Alumno:** Rafael Valerio Villalba

## Formato del CSV

Tu archivo `data/partidas.csv` debe tener **minimo** estas columnas:

| Columna | Descripcion | Ejemplo |
|---------|-------------|---------|
| `numero_ronda` | Numero de la ronda (1, 2, 3...) | 1 |
| `jugada_j1` | Jugada del jugador 1 | piedra |
| `jugada_j2` | Jugada del jugador 2 (oponente) | papel |

### Ejemplo de CSV minimo:

```csv
numero_ronda,jugada_j1,jugada_j2
1,piedra,papel
2,tijera,piedra
3,papel,papel
4,piedra,tijera
...
```

---

## Como recogiste los datos?

Marca con [x] el metodo usado y describe brevemente:

### Metodo de recogida:

- [X] **Programa propio**: Cree un programa para jugar y guardar datos
- [X] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [ ] **Otro**: _________________

### Descripcion del proceso:

```
Creé una página offline mediante HTML, CSS y JavaScript para recoger partidas de piedra-papel-tijera en un ordenador. En ella, los jugadores usaban las teclas 
(Jugador 1: A/S/D, Jugador 2: 4/5/6), cada ronda ocultaba la elección hasta mostrar un popup con el resultado, las rondas 
válidas se guardaban en memoria y al finalizar se descargaba un CSV donde cada fila era una ronda con: timestamp,
 número de ronda, nombres, tecla y código de movimiento (0=piedra, 1=papel, 2=tijera) para cada jugador y el ganador.

Al final decidimos jugar en persona y recoger los datos manualmente porque al probar con el teclado detectamos un sesgo: 
tendemos a pulsar más fácilmente con el índice y menos con el anular por comodidad y fuerza en las manos, lo que 
distorsionaba los datos.

```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [ ] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [x] `sesion` - ID de sesion de juego
- [ ] `resultado` - victoria/derrota/empate
- [ ] Otro: 

### Descripcion de datos adicionales:

```
La columna "Partida" lo cuento como sesión, ya que recogimos partida y número de ronda. De esta manera no "reseteabamos"
cada vez que empatabamos, sino que una partida no estaba ganada hasta que uno no ganaba la ronda.

```

---

## Estadisticas del dataset

- **Total de rondas:** 308
- **Numero de sesiones/partidas:** 200
- **Contra cuantas personas diferentes:** 1 recogido en 4-5 días distintos y haciendo descanso cada 20 partidas.

### Tipo de IA:

- [X] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: Victor Lacruz Sancho
- [ ] **IA General**: Entrenada para ganar a cualquier oponente
