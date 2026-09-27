"""Dino Offline · Algoritmo Genético
======================================

Descripción del problema
------------------------
Una población de dinosaurios aprende a esquivar cactus evolucionando su
cromosoma mediante un Algoritmo Genético (AG). El juego replica las mecánicas
del Dino de Chrome: velocidad creciente, obstáculos aleatorios (cactus pequeños
y grandes, en grupos de 1 a 3) y física de salto con gravedad.

El objetivo es optimizar los parámetros de decisión de salto del dinosaurio
para maximizar la distancia recorrida antes de chocar (función objetivo).

Flujo del Algoritmo Genético (igual al visto en clase)
-------------------------------------------------------
    1. Definir población  →  cromosomas aleatorios dentro del rango de cada gen
    2. Evaluar            →  correr la simulación; aptitud = score (distancia)
    3. Cruzar             →  selección por torneo + cruce de un punto
    4. Mutar              →  perturbación gaussiana sobre un gen aleatorio
    5. Actualizar         →  élite intacta + hijos cruzados y mutados
    6. ¿Criterio parada?  →  max generaciones O convergencia en META_SCORE
       └─ No → volver a 2 (nueva generación)
       └─ Sí → guardar resultados y detener

Cromosoma (4 genes reales)
--------------------------
    F1  Distancia base   [-200, 500] px  — umbral fijo de reacción al obstáculo
    F2  Anticipación     [0, 80]         — escala la distancia con la velocidad
    F3  Fuerza de salto  [6.0, 12.5]     — velocidad vertical inicial al saltar
    F4  Factor de ancho  [-3.0, 3.0]     — ajusta según el ancho del obstáculo

Regla de salto:
    El dino salta cuando:
        hueco_al_obstáculo  ≤  F1 + F2 * velocidad + F4 * ancho_obstáculo

Función objetivo
----------------
    score = int( min(distancia, distancia_meta) * 0.025 )
    Meta: score ≥ 1000  (equivale a recorrer META_DIST px sin chocar)

Persistencia
------------
    Al pausar, terminar o cerrar, los resultados se guardan en:
        resultados_dino_ga.json
    con el historial por generación, el mejor cromosoma y los parámetros usados.

Uso
---
    python algoritmogenetico_dino_aprendizaje.py
    (requiere Python 3 con Tkinter; no necesita dependencias externas)
"""

import json
import math
import os
import random
import socket
import time
import tkinter as tk
from tkinter import ttk

# ── Parámetros iniciales del algoritmo ──────────────────────────────────────
# Estos valores pueden ajustarse desde la interfaz gráfica en tiempo de ejecución.
POBLACION = 20            # Número de individuos (cromosomas) por generación
ELITE = 2                 # Los N mejores individuos pasan intactos (elitismo)
TORNEO = 3                # Tamaño del torneo para selección de padres
PROB_CRUCE = 0.80         # Probabilidad de que ocurra cruce entre dos padres
PROB_MUTACION = 0.75      # Probabilidad de que un hijo sufra mutación
TOP_INTENTOS = 10         # Filas de la tabla de mejores intentos en la GUI
MAX_GENERACIONES = 100    # Criterio de parada 1: límite de generaciones
META_SCORE = 1000         # Criterio de parada 2: score objetivo de convergencia
RACHA_CONVERGENCIA = 2    # Generaciones consecutivas en META_SCORE para converger

# Genes: (nombre, valor_mínimo, valor_máximo)
# Cada tupla define el nombre descriptivo y el rango válido de ese gen.
# Los valores se muestrean uniformemente en [mínimo, máximo] en la población inicial.
GENES = [
    ("Distancia base", -200.0, 500.0),   # F1: umbral fijo de distancia al obstáculo
    ("Anticipación",     0.0,  80.0),    # F2: factor proporcional a la velocidad actual
    ("Fuerza de salto",  6.0,  12.5),    # F3: velocidad vertical inicial del salto
    ("Factor de ancho", -3.0,   3.0),    # F4: ajuste según el ancho del obstáculo
]

# ── Físicas del juego (mismos valores base que el dino de Chrome) ───────────
# Estas constantes replican el motor físico original del juego para que la
# simulación sea fiel al comportamiento real del dinosaurio en el navegador.
W, H = 880, 240          # Dimensiones del canvas (ancho x alto) en píxeles
GROUND = 196             # Coordenada Y del suelo (píxeles desde la parte superior)
DINO_X = 70              # Posición horizontal fija del dinosaurio (no se mueve)
DINO_W = 44              # Ancho del dinosaurio en píxeles
HIT_L, HIT_R = 8, 6       # recorte lateral de la caja de colisión del dino
HIT_MARGIN = 0            # tolerancia vertical
GRAVITY = 0.6
SPEED0, SPEED_MAX, ACCEL = 6.0, 13.0, 0.002
SCORE_COEF = 0.025
META_DIST = META_SCORE / SCORE_COEF
SMALL_W, SMALL_H = 18, 34
BIG_W, BIG_H = 26, 50
MIN_GAP = 120

# ── Estilo ──────────────────────────────────────────────────────────────────
FONT, MONO = "Segoe UI", "Consolas"
BG, CARD, LINE, LINE2 = "#f4f4f2", "#ffffff", "#e7e5e4", "#d6d3d1"
TEXT, MUTED, FAINT, DARK = "#1c1917", "#78716c", "#a8a29e", "#1c1917"
INK, GHOST, GREEN, BLUE = "#535353", "#cfcfcf", "#16a34a", "#2563eb"
FRAME_MS = 16

SPEEDS = [("1x", 1), ("2x", 2), ("4x", 4), ("8x", 8), ("Turbo", 40)]
RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_dino_ga.json")

# ── Sprites (pixel art, cada # se dibuja como 2x2 px) ───────────────────────
DINO_BODY = [
    "...........#########..",
    "..........###########.",
    "..........##.#########",
    "..........###########.",
    "..........###########.",
    "..........###########.",
    "..........#########...",
    "..........#####.......",
    "#.........#####.......",
    "#........######.......",
    "#.......#######...#...",
    "##......#########.##..",
    "###.....##########.#..",
    "####...############...",
    "###################...",
    ".#####################",
    "..###################.",
    "...#################..",
    "....###############...",
    ".....#############....",
]
DINO_LEGS = {
    "stand": ["......####..####......", "......####..####......", "......##....##........",
              "......##....##........", "......####..####......"],
    "run0": ["......####..####......", "......####..##........", "......##....####......",
             "......##..............", "......####............"],
    "run1": ["......####..####......", "........##..####......", "......####..##........",
             "............##........", "............####......"],
}
CACTUS_SMALL = [
    "...###...", "...###...", "...###...", ".#.###...", ".#.###.#.", ".#.###.#.",
    ".#.###.#.", ".#.###.#.", ".#####.#.", ".#####.#.", "...#####.", "...###...",
    "...###...", "...###...", "...###...", "...###...", "...###...",
]
CACTUS_BIG = [
    "....###....", "....###....", "....###....", "....###....", "....###....",
    ".##.###....", ".##.###....", ".##.###....", ".##.###.##.", ".##.###.##.",
    ".##.###.##.", ".##.###.##.", ".##.###.##.", ".##.###.##.", ".######.##.",
    ".######.##.", "....######.", "....######.", "....###....", "....###....",
    "....###....", "....###....", "....###....", "....###....", "....###....",
]


def clamp(v, lo, hi):
    """Restringe el valor `v` al intervalo [lo, hi].

    Usado para mantener los genes dentro de su rango válido tras la mutación
    y para normalizar fracciones en los deslizadores de la interfaz.

    Args:
        v:  Valor a recortar.
        lo: Límite inferior del intervalo.
        hi: Límite superior del intervalo.

    Returns:
        El valor v acotado: max(lo, min(v, hi)).
    """
    return max(lo, min(v, hi))


def score_de(distancia):
    """Convierte una distancia en píxeles al puntaje mostrado en pantalla.

    Implementa la función objetivo del problema:
        score = int( min(distancia, META_DIST) × SCORE_COEF )
    donde SCORE_COEF = 0.025, de modo que META_DIST píxeles equivalen a
    exactamente META_SCORE = 1000 puntos.

    Args:
        distancia: Distancia total recorrida por el dinosaurio (px).

    Returns:
        Puntaje entero en el rango [0, META_SCORE].
    """
    return int(min(distancia, META_DIST) * SCORE_COEF)


# ════════════════════════════════════════════════════════════════════════════
#  ALGORITMO GENÉTICO
# ════════════════════════════════════════════════════════════════════════════
class AlgoritmoGenetico:
    """Implementa el ciclo completo del Algoritmo Genético para optimizar
    los parámetros de salto del dinosaurio.

    Sigue el flujo canónico enseñado en clase:
        Definir población → Evaluar → Cruzar → Mutar → Actualizar → ¿Parar?

    Atributos:
        p_cruce (float):     Probabilidad de cruce entre dos padres [0, 1].
        p_mutacion (float):  Probabilidad de mutar un hijo [0, 1].
        generacion (int):    Número de generación actual (empieza en 1).
        poblacion (list):    Lista de cromosomas; ordenada de mejor a peor
                             tras cada llamada a evaluar().
        aptitud (list):      Lista de aptitudes (scores) paralela a poblacion.
        mejor (tuple|None):  (aptitud, cromosoma) del mejor individuo histórico.
    """

    def __init__(self):
        """Inicializa el AG con los parámetros globales y genera la población."""
        self.p_cruce = PROB_CRUCE
        self.p_mutacion = PROB_MUTACION
        self.generacion = 1
        self.poblacion = []      # cromosomas; tras evaluar quedan ordenados de mejor a peor
        self.aptitud = []        # aptitud de cada cromosoma (misma posición)
        self.mejor = None        # (aptitud, cromosoma) récord histórico
        self.definir_poblacion()

    # 1. Definir población: cromosomas aleatorios dentro del rango de cada gen
    def definir_poblacion(self):
        """Paso 1 — Genera la población inicial de forma aleatoria.

        Cada cromosoma es una lista de 4 genes (F1, F2, F3, F4) cuyos valores
        se muestrean con distribución uniforme dentro del rango [min, max]
        definido en la constante GENES. Reinicia también la lista de aptitudes.
        """
        self.poblacion = [[random.uniform(lo, hi) for _, lo, hi in GENES] for _ in range(POBLACION)]
        self.aptitud = []

    # 2. Evaluar: la aptitud la calcula la simulación (función objetivo)
    def evaluar(self, aptitudes):
        """Paso 2 — Registra las aptitudes y ordena la población de mejor a peor.

        Recibe la lista de scores producida por la simulación (una por individuo),
        empareja cada score con su cromosoma, ordena de mayor a menor aptitud y
        actualiza el récord histórico si el mejor de esta generación lo supera.

        Args:
            aptitudes (list[int]): Lista de scores (función objetivo) en el mismo
                                   orden que self.poblacion antes de ordenar.
        """
        ranking = sorted(zip(aptitudes, self.poblacion), key=lambda p: p[0], reverse=True)
        self.aptitud = [a for a, _ in ranking]
        self.poblacion = [g for _, g in ranking]
        if self.mejor is None or self.aptitud[0] >= self.mejor[0]:
            self.mejor = (self.aptitud[0], self.poblacion[0][:])

    def _torneo(self):
        """Selección por torneo: elige al mejor entre TORNEO candidatos aleatorios.

        Selecciona TORNEO índices al azar de la población ordenada. Como la lista
        ya está ordenada de mayor a menor aptitud, el índice más pequeño corresponde
        al individuo más apto. Devuelve una copia de ese cromosoma.

        Returns:
            list[float]: Cromosoma ganador del torneo (copia de la lista).
        """
        candidatos = random.sample(range(len(self.poblacion)), TORNEO)
        return self.poblacion[min(candidatos)]   # el índice menor es el mejor

    # 3. Cruzar: cruce de un punto con probabilidad p_cruce
    def cruzar(self):
        """Paso 3 — Genera POBLACION - ELITE hijos mediante cruce de un punto.

        Para cada hijo:
          1. Se seleccionan dos padres A y B independientemente por torneo.
          2. Con probabilidad p_cruce se elige un punto de corte aleatorio
             entre las posiciones 1 y len(GENES)-1 del cromosoma.
          3. El hijo hereda los genes [0:punto] de A y los genes [punto:] de B.
          4. Sin cruce (prob. 1 - p_cruce), el hijo es copia directa del padre A.

        Returns:
            list[list[float]]: Lista de (POBLACION - ELITE) cromosomas hijos.
        """
        hijos = []
        while len(hijos) < POBLACION - ELITE:
            a, b = self._torneo(), self._torneo()
            if random.random() < self.p_cruce:
                punto = random.randint(1, len(GENES) - 1)
                hijos.append(a[:punto] + b[punto:])
            else:
                hijos.append(a[:])
        return hijos

    # 4. Mutar: con probabilidad p_mutacion cambia un gen al azar
    def mutar(self, hijos):
        """Paso 4 — Aplica mutación gaussiana a los hijos con probabilidad p_mutacion.

        Para cada hijo, con probabilidad p_mutacion:
          1. Se elige un gen aleatorio i entre 0 y len(GENES)-1.
          2. Se le suma un ruido gaussiano: gen[i] += Gauss(0, (hi-lo) * 0.15)
          3. El resultado se recorta al rango [lo, hi] del gen para mantener validez.

        La desviación estándar del 15% del rango permite explorar el vecindario
        de la solución actual sin destruir completamente los valores buenos.

        Args:
            hijos (list[list[float]]): Lista de cromosomas hijos (se modifican in-place).
        """
        for hijo in hijos:
            if random.random() < self.p_mutacion:
                i = random.randrange(len(GENES))
                _, lo, hi = GENES[i]
                hijo[i] = clamp(hijo[i] + random.gauss(0, (hi - lo) * 0.15), lo, hi)

    # 5. Actualizar población: élite + hijos
    def actualizar(self, hijos):
        """Paso 5 — Construye la nueva generación combinando élite e hijos.

        Los ELITE mejores individuos de la generación actual (posiciones 0 a
        ELITE-1, ya que la lista está ordenada) se copian intactos. Los
        (POBLACION - ELITE) hijos generados por cruce y mutación completan la
        nueva población. Incrementa el contador de generación.

        Args:
            hijos (list[list[float]]): Hijos generados por cruzar() y modificados
                                       por mutar().
        """
        elite = [g[:] for g in self.poblacion[:ELITE]]
        self.poblacion = elite + hijos
        self.aptitud = []
        self.generacion += 1


# ════════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN DEL JUEGO (la función objetivo)
# ════════════════════════════════════════════════════════════════════════════
class Corredor:
    """Representa a un dinosaurio individual durante la simulación.

    Encapsula el estado físico y de vida de un individuo en el contexto del
    juego. El atributo `genes` contiene el cromosoma [F1, F2, F3, F4] que
    determina su estrategia de salto.

    Atributos:
        genes (list[float]):  Cromosoma del individuo [F1, F2, F3, F4].
        y (float):            Altura actual sobre el suelo (0 = en tierra).
        vy (float):           Velocidad vertical actual (px/frame).
        aire (bool):          True si el dinosaurio está en el aire.
        vivo (bool):          False cuando ha chocado con un obstáculo.
        distancia (float):    Distancia total recorrida al momento de morir
                              (o META_DIST si sobrevivió hasta el final).
    """
    __slots__ = ("genes", "y", "vy", "aire", "vivo", "distancia")

    def __init__(self, genes):
        """Inicializa un corredor en posición inicial (en tierra, vivo).

        Args:
            genes (list[float]): Cromosoma [F1, F2, F3, F4] del individuo.
        """
        self.genes = genes
        self.y = 0.0
        self.vy = 0.0
        self.aire = False
        self.vivo = True
        self.distancia = 0.0


class Simulacion:
    """Ejecuta la función objetivo: corre todos los individuos simultáneamente.

    Todos los corredores comparten el mismo recorrido (misma semilla aleatoria
    para los obstáculos) para que las diferencias de aptitud reflejen únicamente
    la calidad del cromosoma y no la aleatoriedad del entorno.

    La simulación avanza frame a frame mediante el método paso(). En cada frame:
      - La velocidad del mundo aumenta gradualmente.
      - Se generan nuevos obstáculos al vuelo cuando es necesario.
      - Cada corredor aplica su regla de salto (genes) y actualiza su física.
      - Se detectan colisiones; el corredor muere y se registra su distancia.
      - La simulación termina cuando todos han muerto o se alcanza META_DIST.

    Atributos:
        corredores (list[Corredor]): Todos los individuos de la generación.
        distancia (float):  Distancia total recorrida por el mundo (px).
        velocidad (float):  Velocidad actual del desplazamiento (px/frame).
        vivos (int):        Número de corredores aún en pie.
        terminada (bool):   True cuando la simulación ha concluido.
        lider (Corredor):   El último corredor en morir (o el primero en llegar).
    """

    def __init__(self, cromosomas, rng=None):
        """Inicializa la simulación con una población de cromosomas.

        Args:
            cromosomas (list[list[float]]): Lista de cromosomas a evaluar.
            rng (random.Random, optional):  Generador aleatorio independiente
                para los obstáculos. Si es None se crea uno nuevo.
        """
        self.rng = rng or random.Random()
        self.distancia = 0.0
        self.velocidad = SPEED0
        self.frame = 0
        self.obstaculos = []
        self.sig_x = 520.0
        self.corredores = [Corredor(g) for g in cromosomas]
        self.vivos = len(self.corredores)
        self.terminada = False
        self.lider = self.corredores[0]     # último en morir
        self._llenar()

    def _llenar(self):
        """Genera obstáculos hacia adelante hasta tener al menos W+120 px de margen.

        Cada obstáculo se crea con tipo aleatorio (grande/pequeño), número de
        cactus agrupados (1-3) y separación proporcional a la velocidad actual.
        """
        while self.sig_x < self.distancia + W + 120:
            grande = self.rng.random() < 0.5
            n = self.rng.choice((1, 1, 2, 3))
            ancho1, alto = (BIG_W, BIG_H) if grande else (SMALL_W, SMALL_H)
            ancho = ancho1 * n
            self.obstaculos.append({"x": self.sig_x, "w": ancho, "h": alto, "w1": ancho1, "n": n, "big": grande})
            self.sig_x += ancho + round(ancho * self.velocidad + MIN_GAP * self.rng.uniform(1.0, 1.5))

    def paso(self):
        """Avanza la simulación un frame (≈16 ms a 60 fps).

        En cada frame se realizan las siguientes operaciones en orden:
          1. Incrementar el frame y actualizar la velocidad del mundo.
          2. Avanzar la distancia total recorrida.
          3. Generar nuevos obstáculos si es necesario.
          4. Eliminar obstáculos que ya quedaron fuera de pantalla por la izquierda.
          5. Identificar el obstáculo más próximo al dinosaurio.
          6. Para cada corredor vivo:
             a. Aplicar la regla de salto si se cumple la condición del cromosoma.
             b. Actualizar la física del salto (posición y velocidad vertical).
             c. Detectar colisión con el obstáculo próximo; matar al corredor si choca.
          7. Verificar si la simulación ha terminado (todos muertos o META_DIST alcanzada).
        """
        self.frame += 1
        self.velocidad = min(SPEED_MAX, self.velocidad + ACCEL)
        self.distancia += self.velocidad
        self._llenar()
        d = self.distancia
        while self.obstaculos and self.obstaculos[0]["x"] - d + self.obstaculos[0]["w"] < 0:
            self.obstaculos.pop(0)

        obs, sx, hueco = None, 0.0, 0.0
        for o in self.obstaculos:
            if o["x"] - d + o["w"] > DINO_X + HIT_L:
                obs = o
                sx = o["x"] - d
                hueco = sx - (DINO_X + DINO_W)
                break

        for r in self.corredores:
            if not r.vivo:
                continue
            g = r.genes
            # decisión: saltar cuando el obstáculo entra en la distancia de reacción
            if obs is not None and not r.aire and hueco <= g[0] + g[1] * self.velocidad + g[3] * obs["w"]:
                r.aire = True
                r.vy = g[2]
            if r.aire:
                r.y += r.vy
                r.vy -= GRAVITY
                if r.y <= 0:
                    r.y, r.vy, r.aire = 0.0, 0.0, False
            if (obs is not None and r.y < obs["h"] - HIT_MARGIN
                    and sx < DINO_X + DINO_W - HIT_R and sx + obs["w"] > DINO_X + HIT_L):
                r.vivo = False
                r.distancia = d
                self.vivos -= 1
                self.lider = r

        if self.vivos == 0:
            self.terminada = True
        elif d >= META_DIST:
            for r in self.corredores:
                if r.vivo:
                    r.distancia = META_DIST
            self.lider = next(r for r in self.corredores if r.vivo)
            self.terminada = True

    def aptitudes(self):
        """Calcula y devuelve la aptitud (score) de cada corredor.

        Aplica la función objetivo score_de(distancia) a cada individuo.
        Esta lista se pasa directamente a AlgoritmoGenetico.evaluar().

        Returns:
            list[int]: Lista de scores en el mismo orden que self.corredores.
        """
        return [score_de(r.distancia) for r in self.corredores]


# ════════════════════════════════════════════════════════════════════════════
#  WIDGETS
# ════════════════════════════════════════════════════════════════════════════
def crear_sprite(filas, color, zoom=2):
    h, w = len(filas), len(filas[0])
    img = tk.PhotoImage(width=w, height=h)
    for y, fila in enumerate(filas):
        x = 0
        while x < w:
            if fila[x] == "#":
                x0 = x
                while x < w and fila[x] == "#":
                    x += 1
                img.put(color, to=(x0, y, x, y + 1))
            else:
                x += 1
    return img.zoom(zoom, zoom)


class Boton(tk.Label):
    def __init__(self, master, texto, comando, primario=True):
        self.primario = primario
        self.comando = comando
        super().__init__(master, text=texto, font=(FONT, -13, "bold"), cursor="hand2", padx=20, pady=9,
                         highlightthickness=1)
        self._pintar(False)
        self.bind("<Enter>", lambda e: self._pintar(True))
        self.bind("<Leave>", lambda e: self._pintar(False))
        self.bind("<Button-1>", lambda e: self.comando())

    def _pintar(self, hover):
        if self.primario:
            self.config(bg="#3b3835" if hover else DARK, fg="#ffffff", highlightbackground=DARK)
        else:
            self.config(bg="#f5f5f4" if hover else CARD, fg=TEXT, highlightbackground=LINE2)


class Deslizador(tk.Canvas):
    def __init__(self, master, lo, hi, valor, paso, on_change):
        super().__init__(master, height=20, bg=CARD, highlightthickness=0, cursor="hand2")
        self.lo, self.hi, self.paso, self.valor, self.on_change = lo, hi, paso, valor, on_change
        self.bind("<Configure>", lambda e: self._dibujar())
        self.bind("<Button-1>", self._mover)
        self.bind("<B1-Motion>", self._mover)

    def _dibujar(self):
        self.delete("all")
        w, pad, cy = self.winfo_width(), 8, 10
        x = pad + (self.valor - self.lo) / (self.hi - self.lo) * (w - 2 * pad)
        self.create_line(pad, cy, w - pad, cy, fill=LINE, width=4, capstyle="round")
        self.create_line(pad, cy, x, cy, fill=DARK, width=4, capstyle="round")
        self.create_oval(x - 7, cy - 7, x + 7, cy + 7, fill="#ffffff", outline=DARK, width=2)

    def _mover(self, e):
        w, pad = self.winfo_width(), 8
        t = clamp((e.x - pad) / (w - 2 * pad), 0, 1)
        v = round((self.lo + t * (self.hi - self.lo)) / self.paso) * self.paso
        self.valor = round(clamp(v, self.lo, self.hi), 4)
        self._dibujar()
        self.on_change(self.valor)


class Segmentado(tk.Frame):
    def __init__(self, master, opciones, valor, on_change):
        super().__init__(master, bg="#f0efed", padx=2, pady=2)
        self.on_change = on_change
        self.items = {}
        for texto, val in opciones:
            lb = tk.Label(self, text=texto, font=(FONT, -12, "bold"), cursor="hand2", pady=5)
            lb.pack(side="left", fill="x", expand=True)
            lb.bind("<Button-1>", lambda e, v=val: self.seleccionar(v))
            self.items[val] = lb
        self.seleccionar(valor, avisar=False)

    def seleccionar(self, valor, avisar=True):
        for v, lb in self.items.items():
            activo = v == valor
            lb.config(bg=CARD if activo else "#f0efed", fg=TEXT if activo else MUTED)
        if avisar:
            self.on_change(valor)


def tarjeta(master):
    return tk.Frame(master, bg=CARD, highlightbackground=LINE, highlightthickness=1, bd=0)


def etiqueta(master, texto, tam=-13, peso="normal", fg=TEXT, bg=CARD, fuente=FONT, **kw):
    return tk.Label(master, text=texto, font=(fuente, tam, peso), fg=fg, bg=bg, **kw)


def titulo_seccion(master, texto):
    return etiqueta(master, texto, -11, "bold", MUTED)


# ════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN
# ════════════════════════════════════════════════════════════════════════════
class App:
    """Controlador principal de la aplicación: integra el AG, la simulación y la GUI.

    Responsabilidades:
      - Construir y gestionar todos los widgets de la interfaz gráfica (Tkinter).
      - Orquestar el ciclo del AG: población → evaluar → cruzar → mutar →
        actualizar → criterio de parada, mostrando cada paso visualmente.
      - Animar la simulación del juego en el canvas a velocidades configurables.
      - Actualizar en tiempo real la gráfica de aptitud, la tabla de mejores
        intentos y el panel de estado.
      - Persistir los resultados automáticamente en resultados_dino_ga.json.

    La aplicación corre en el hilo principal de Tkinter; el loop de animación
    se implementa con root.after() para no bloquear la interfaz.

    Constantes de layout:
        SIDE_W (int):   Ancho del panel lateral en píxeles.
        CHART_W (int):  Ancho de la gráfica de aptitud.
        CHART_H (int):  Alto de la gráfica de aptitud.
    """
    SIDE_W = 300
    CHART_W, CHART_H = 556, 238

    def __init__(self, root):
        """Inicializa la aplicación: crea el AG, la simulación y construye la GUI.

        Args:
            root (tk.Tk): Ventana raíz de Tkinter.
        """
        self.root = root
        root.title("Dino · Algoritmo Genético")
        root.configure(bg=BG)
        root.resizable(False, False)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        ww, wh = 1252, min(810, sh - 70)
        root.geometry(f"{ww}x{wh}+{max(0, (sw - ww) // 2)}+{max(0, (sh - wh) // 2 - 20)}")

        self.ga = AlgoritmoGenetico()
        self.sim = None
        self.historial = []
        self.record = 0
        self.racha = 0
        self.max_gen = MAX_GENERACIONES
        self.velocidad = 4
        self.running = False
        self.finished = False
        self.phase = "idle"
        self.cola = []
        self.espera = 0.0
        self.acum = 0.0
        self.t_prev = 0.0
        self.hijos = []
        self.after_id = None
        self.dead_shown = None
        self._cache_stats = None

        self._cargar_sprites()
        self._estilo_tabla()
        self._construir()
        self._nueva_escena()
        self._set_flow(None)
        self._set_status("idle", "Listo")
        self._actualizar_cromosoma()
        self._dibujar_grafica()
        self._actualizar_stats()
        self._pintar()
        root.protocol("WM_DELETE_WINDOW", self._cerrar)

    # ── sprites ─────────────────────────────────────────────────────────────
    def _cargar_sprites(self):
        self.img = {}
        muerto = DINO_BODY[:]
        muerto[2], muerto[3] = "..........##..########", "..........##..########"
        for clave, color in (("dark", INK), ("ghost", GHOST)):
            poses = {p: crear_sprite(DINO_BODY + legs, color) for p, legs in DINO_LEGS.items()}
            poses["dead"] = crear_sprite(muerto + DINO_LEGS["stand"], color)
            poses["small"] = crear_sprite(CACTUS_SMALL, INK)
            poses["big"] = crear_sprite(CACTUS_BIG, INK)
            self.img[clave] = poses
        rng = random.Random(7)
        self.bumps = [(rng.uniform(0, 1600), rng.randint(6, 14), rng.randint(2, 9)) for _ in range(70)]
        self.nubes = [(120, 40), (430, 66), (720, 34), (1000, 56)]

    def _estilo_tabla(self):
        st = ttk.Style()
        st.theme_use("clam")
        st.configure("Dino.Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT,
                     borderwidth=0, rowheight=25, font=(FONT, -12))
        st.configure("Dino.Treeview.Heading", background=CARD, foreground=MUTED, borderwidth=0,
                     relief="flat", font=(FONT, -11, "bold"))
        st.map("Dino.Treeview.Heading", background=[("active", CARD)])
        st.map("Dino.Treeview", background=[("selected", "#f0efed")], foreground=[("selected", TEXT)])
        st.layout("Dino.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])

    # ── layout ──────────────────────────────────────────────────────────────
    def _construir(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=24, pady=20)

        # cabecera
        head = tk.Frame(outer, bg=BG)
        head.pack(fill="x", pady=(0, 16))
        izq = tk.Frame(head, bg=BG)
        izq.pack(side="left")
        etiqueta(izq, "Dino  ·  Algoritmo Genético", -22, "bold", bg=BG).pack(anchor="w")
        etiqueta(izq, f"Población de {POBLACION} individuos  ·  cromosoma de {len(GENES)} genes  ·  "
                      f"meta de {META_SCORE} puntos", -12, fg=MUTED, bg=BG).pack(anchor="w", pady=(2, 0))
        der = tk.Frame(head, bg=BG)
        der.pack(side="right")
        self.status = tk.Label(der, font=(FONT, -12, "bold"), padx=14, pady=7)
        self.status.pack(side="left", padx=(0, 14))
        self.btn_reset = Boton(der, "Reiniciar", self.reiniciar, primario=False)
        self.btn_reset.pack(side="left", padx=(0, 10))
        self.btn_main = Boton(der, "Iniciar entrenamiento", self.alternar)
        self.btn_main.pack(side="left")

        body = tk.Frame(outer, bg=BG)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y")
        right = tk.Frame(body, bg=BG, width=self.SIDE_W)
        right.pack(side="left", fill="y", padx=(20, 0))
        right.pack_propagate(False)

        self._construir_juego(left)
        self._construir_flujo(left)
        self._construir_inferior(left)
        self._construir_lateral(right)

    def _construir_juego(self, parent):
        card = tarjeta(parent)
        card.pack(fill="x")
        self.canvas = tk.Canvas(card, width=W, height=H, bg=CARD, highlightthickness=0)
        self.canvas.pack()
        # sprites de los individuos: primero los fantasmas, al final el líder (queda encima)
        self.items = [None] * POBLACION
        for i in reversed(range(POBLACION)):
            self.items[i] = self.canvas.create_image(DINO_X, GROUND, anchor="sw", state="hidden",
                                                     image=self.img["dark" if i == 0 else "ghost"]["stand"])

    def _construir_flujo(self, parent):
        fila = tk.Frame(parent, bg=BG)
        fila.pack(fill="x", pady=(12, 12))
        self.chips = {}
        pasos = [("poblacion", "Población"), ("evaluar", "Evaluar"), ("cruzar", "Cruzar"),
                 ("mutar", "Mutar"), ("actualizar", "Actualizar"), ("parada", "¿Parar?")]
        for i, (clave, texto) in enumerate(pasos):
            if i:
                etiqueta(fila, "›", -15, fg=FAINT, bg=BG).pack(side="left", padx=6)
            lb = tk.Label(fila, text=texto, font=(FONT, -12, "bold"), padx=13, pady=5)
            lb.pack(side="left")
            self.chips[clave] = lb
        self.nota = etiqueta(fila, "", -12, fg=MUTED, bg=BG)
        self.nota.pack(side="right")

    def _construir_inferior(self, parent):
        fila = tk.Frame(parent, bg=BG)
        fila.pack(fill="both", expand=True)

        graf = tarjeta(fila)
        graf.pack(side="left", fill="y")
        top = tk.Frame(graf, bg=CARD)
        top.pack(fill="x", padx=16, pady=(14, 0))
        titulo_seccion(top, "APTITUD POR GENERACIÓN").pack(side="left")
        for texto, color in (("Promedio", FAINT), ("Mejor", DARK)):
            etiqueta(top, texto, -11, fg=MUTED).pack(side="right")
            tk.Frame(top, bg=color, width=14, height=3).pack(side="right", padx=(12, 5))
        self.chart = tk.Canvas(graf, width=self.CHART_W, height=self.CHART_H, bg=CARD, highlightthickness=0)
        self.chart.pack(padx=8, pady=(4, 8))

        hist = tarjeta(fila)
        hist.pack(side="left", fill="both", expand=True, padx=(16, 0))
        titulo_seccion(hist, f"TOP {TOP_INTENTOS} INTENTOS").pack(anchor="w", padx=16, pady=(14, 6))
        self.tabla = ttk.Treeview(hist, columns=("puesto", "gen", "mejor", "prom"), show="headings",
                                  style="Dino.Treeview", selectmode="none", height=TOP_INTENTOS)
        for col, texto, ancho in (("puesto", "#", 34), ("gen", "Intento", 62), ("mejor", "Score", 66),
                                  ("prom", "Promedio", 78)):
            self.tabla.heading(col, text=texto)
            self.tabla.column(col, width=ancho, anchor="center")
        self.tabla.tag_configure("meta", foreground=GREEN)
        self.tabla.tag_configure("actual", background="#eff6ff")
        self.tabla.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _construir_lateral(self, parent):
        # Estado
        est = tarjeta(parent)
        est.pack(fill="x")
        interior = tk.Frame(est, bg=CARD)
        interior.pack(fill="x", padx=16, pady=14)
        titulo_seccion(interior, "ESTADO").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.stat = {}
        celdas = [("gen", "GENERACIÓN"), ("vivos", "VIVOS"), ("record", "RÉCORD"), ("conv", "CONVERGENCIA")]
        for i, (clave, texto) in enumerate(celdas):
            r, c = 1 + (i // 2) * 2, i % 2
            etiqueta(interior, texto, -10, "bold", FAINT).grid(row=r, column=c, sticky="w", pady=(4, 0))
            lb = etiqueta(interior, "–", -22, "bold")
            lb.grid(row=r + 1, column=c, sticky="w")
            self.stat[clave] = lb
        interior.grid_columnconfigure(0, weight=1)
        interior.grid_columnconfigure(1, weight=1)
        self.progreso = tk.Canvas(interior, height=6, bg=CARD, highlightthickness=0)
        self.progreso.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.progreso.bind("<Configure>", lambda e: self._dibujar_progreso())

        # Parámetros
        par = tarjeta(parent)
        par.pack(fill="x", pady=(12, 0))
        interior = tk.Frame(par, bg=CARD)
        interior.pack(fill="x", padx=16, pady=14)
        titulo_seccion(interior, "PARÁMETROS").pack(anchor="w", pady=(0, 6))

        def control(texto, lo, hi, valor, paso, fmt, on_change):
            fila = tk.Frame(interior, bg=CARD)
            fila.pack(fill="x", pady=(6, 0))
            etiqueta(fila, texto, -12, fg=MUTED).pack(side="left")
            lb = etiqueta(fila, fmt(valor), -12, "bold", fuente=MONO)
            lb.pack(side="right")
            Deslizador(interior, lo, hi, valor, paso,
                       lambda v: (lb.config(text=fmt(v)), on_change(v))).pack(fill="x")

        control("Probabilidad de cruce", 0, 1, PROB_CRUCE, 0.05, lambda v: f"{v:.2f}",
                lambda v: setattr(self.ga, "p_cruce", v))
        control("Probabilidad de mutación", 0, 1, PROB_MUTACION, 0.05, lambda v: f"{v:.2f}",
                lambda v: setattr(self.ga, "p_mutacion", v))
        control("Máx. generaciones", 10, 500, MAX_GENERACIONES, 10, lambda v: f"{int(v)}",
                lambda v: setattr(self, "max_gen", int(v)))
        fila = tk.Frame(interior, bg=CARD)
        fila.pack(fill="x", pady=(10, 6))
        etiqueta(fila, "Velocidad de simulación", -12, fg=MUTED).pack(side="left")
        Segmentado(interior, SPEEDS, self.velocidad, lambda v: setattr(self, "velocidad", v)).pack(fill="x")

        # Cromosoma
        cro = tarjeta(parent)
        cro.pack(fill="both", expand=True, pady=(12, 0))
        interior = tk.Frame(cro, bg=CARD)
        interior.pack(fill="x", padx=16, pady=14)
        titulo_seccion(interior, "MEJOR CROMOSOMA").pack(anchor="w", pady=(0, 4))
        self.genes_ui = []
        for i, (nombre, lo, hi) in enumerate(GENES):
            fila = tk.Frame(interior, bg=CARD)
            fila.pack(fill="x", pady=(6, 0))
            etiqueta(fila, f"F{i + 1}", -11, "bold", FAINT, fuente=MONO).pack(side="left")
            etiqueta(fila, nombre, -12, fg=MUTED).pack(side="left", padx=(8, 0))
            val = etiqueta(fila, "–", -12, "bold", fuente=MONO)
            val.pack(side="right")
            barra = tk.Canvas(interior, height=5, bg=CARD, highlightthickness=0)
            barra.pack(fill="x", pady=(3, 0))
            self.genes_ui.append((val, barra))

    # ── estado visual ───────────────────────────────────────────────────────
    def _set_status(self, tipo, texto):
        colores = {"idle": ("#f0efed", MUTED), "run": ("#dbeafe", "#1d4ed8"),
                   "pause": ("#fef3c7", "#92400e"), "done": ("#dcfce7", "#166534")}
        bg, fg = colores[tipo]
        self.status.config(text="●  " + texto, bg=bg, fg=fg)

    def _set_flow(self, activo, nota=""):
        for clave, lb in self.chips.items():
            if clave == activo:
                lb.config(bg=DARK, fg="#ffffff")
            else:
                lb.config(bg="#ebeae7", fg=MUTED)
        self.nota.config(text=nota)

    def _actualizar_botones(self):
        if self.running:
            self.btn_main.config(text="Detener")
        elif self.phase == "idle":
            self.btn_main.config(text="Iniciar entrenamiento")
        else:
            self.btn_main.config(text="Continuar")

    def _actualizar_cromosoma(self):
        genes = self.ga.mejor[1] if self.ga.mejor else self.ga.poblacion[0]
        for (val, barra), g, (_, lo, hi) in zip(self.genes_ui, genes, GENES):
            val.config(text=f"{g:+.2f}" if lo < 0 else f"{g:.2f}")
            barra.delete("all")
            w = barra.winfo_width() if barra.winfo_width() > 1 else self.SIDE_W - 34
            barra.create_rectangle(0, 0, w, 5, fill=LINE, outline="")
            barra.create_rectangle(0, 0, w * (g - lo) / (hi - lo), 5, fill=DARK, outline="")

    def _dibujar_progreso(self):
        c = self.progreso
        c.delete("all")
        w = c.winfo_width()
        c.create_rectangle(0, 0, w, 6, fill=LINE, outline="")
        frac = clamp(self.record / META_SCORE, 0, 1)
        c.create_rectangle(0, 0, w * frac, 6, fill=GREEN if frac >= 1 else DARK, outline="")

    def _actualizar_stats(self):
        vivos = self.sim.vivos
        datos = (self.ga.generacion, self.max_gen, vivos, self.record, self.racha)
        if datos == self._cache_stats:
            return
        self._cache_stats = datos
        self.stat["gen"].config(text=f"{datos[0]}")
        self.stat["vivos"].config(text=f"{datos[2]}/{POBLACION}")
        self.stat["record"].config(text=f"{datos[3]}")
        self.stat["conv"].config(text=f"{datos[4]}/{RACHA_CONVERGENCIA}")
        self._dibujar_progreso()

    def _dibujar_grafica(self):
        c = self.chart
        c.delete("all")
        w, h = self.CHART_W, self.CHART_H
        ml, mr, mt, mb = 44, 14, 12, 26
        pw, ph = w - ml - mr, h - mt - mb
        for v in range(0, META_SCORE + 1, META_SCORE // 4):
            y = mt + ph - ph * v / META_SCORE
            c.create_line(ml, y, w - mr, y, fill=LINE if v else LINE2)
            c.create_text(ml - 8, y, text=str(v), anchor="e", fill=FAINT, font=(FONT, -10))
        n = len(self.historial)
        xmax = max(10, n)
        paso = max(1, math.ceil(xmax / 10))

        def px(g):
            return ml + (g - 1) / (xmax - 1) * pw

        def py(v):
            return mt + ph - ph * min(v, META_SCORE) / META_SCORE

        for g in range(1, xmax + 1, paso):
            c.create_text(px(g), h - mb + 13, text=str(g), fill=FAINT, font=(FONT, -10))
        c.create_line(ml, py(META_SCORE), w - mr, py(META_SCORE), fill=GREEN, dash=(4, 3))
        if not n:
            c.create_text(ml + pw / 2, mt + ph / 2, text="Sin datos todavía · inicia el entrenamiento",
                          fill=FAINT, font=(FONT, -12))
            return
        for clave, color, ancho in (("prom", FAINT, 2), ("mejor", DARK, 2)):
            pts = [(px(e["gen"]), py(e[clave])) for e in self.historial]
            if len(pts) > 1:
                c.create_line(*[v for p in pts for v in p], fill=color, width=ancho, smooth=False,
                              joinstyle="round")
            x, y = pts[-1]
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline=CARD, width=2)

    # ── escena del juego ────────────────────────────────────────────────────
    def _nueva_escena(self):
        self.dead_shown = None
        self.sim = Simulacion(self.ga.poblacion, random.Random(random.getrandbits(32)))

    def _pintar(self, mensaje=None, sub=None):
        c, sim = self.canvas, self.sim
        c.delete("dyn")
        d = sim.distancia

        for bx, by in self.nubes:
            x = (bx - d * 0.15) % (W + 160) - 60
            c.create_line(x, by, x + 44, by, width=9, capstyle="round", fill="#ececea", tags="dyn")
            c.create_line(x + 12, by - 7, x + 34, by - 7, width=9, capstyle="round", fill="#ececea", tags="dyn")

        c.create_line(0, GROUND + 1, W, GROUND + 1, fill=INK, width=2, tags="dyn")
        for bx, off, ln in self.bumps:
            x = (bx - d) % 1600
            if x < W:
                c.create_line(x, GROUND + off, x + ln, GROUND + off, fill=FAINT, width=2, tags="dyn")

        for o in sim.obstaculos:
            x = o["x"] - d
            if x < W + 40:
                img = self.img["dark"]["big" if o["big"] else "small"]
                for k in range(o["n"]):
                    c.create_image(x + k * o["w1"], GROUND + 1, image=img, anchor="sw", tags="dyn")

        pose_paso = "run0" if (sim.frame // 5) % 2 == 0 else "run1"
        for i, (r, item) in enumerate(zip(sim.corredores, self.items)):
            pool = self.img["dark" if i == 0 else "ghost"]
            if r.vivo:
                c.itemconfig(item, state="normal", image=pool["stand"] if r.aire or sim.frame == 0 else pool[pose_paso])
                c.coords(item, DINO_X, GROUND + 1 - r.y)
            elif r is self.dead_shown:
                c.itemconfig(item, state="normal", image=self.img["dark"]["dead"])
                c.coords(item, DINO_X, GROUND + 1 - r.y)
                c.tag_raise(item)
            else:
                c.itemconfig(item, state="hidden")

        c.create_text(W - 24, 26, anchor="e", fill=INK, font=(MONO, -17, "bold"), tags="dyn",
                      text=f"HI {self.record:05d}   {score_de(d):05d}")
        c.create_text(24, 26, anchor="w", fill=FAINT, font=(FONT, -12, "bold"), tags="dyn",
                      text=f"GENERACIÓN {self.ga.generacion}    ·    VIVOS {sim.vivos}/{POBLACION}")
        if mensaje:
            c.create_text(W / 2, 92, text=mensaje, fill=INK, font=(MONO, -24, "bold"), tags="dyn")
        if sub:
            c.create_text(W / 2, 124, text=sub, fill=MUTED, font=(FONT, -13), tags="dyn")

    # ── control ─────────────────────────────────────────────────────────────
    def alternar(self):
        """Alterna entre iniciar y detener el entrenamiento (botón principal)."""
        if self.running:
            self.detener()
        else:
            self.iniciar()

    def iniciar(self):
        """Inicia o reanuda el ciclo de entrenamiento del AG.

        Si es la primera vez (phase == 'idle'), encola los pasos iniciales
        'poblacion' y 'evaluar'. Activa el loop de animación con root.after().
        """
        if self.finished:
            self.finished = False
            self.racha = 0
        if self.ga.generacion > self.max_gen:
            self._set_status("pause", "Sube el máx. de generaciones")
            return
        if self.phase == "idle":
            self.cola = ["poblacion", "evaluar"]
            self.phase = "cola"
            self.espera = 0.0
        self.running = True
        self.t_prev = time.perf_counter()
        self._set_status("run", "Entrenando")
        self._actualizar_botones()
        self._agendar()

    def detener(self):
        """Pausa el entrenamiento y guarda los resultados actuales en el JSON."""
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self._set_status("pause", "Detenido")
        self._actualizar_botones()
        self.guardar()

    def reiniciar(self):
        """Reinicia completamente el AG: nueva población, historial limpio, GUI reseteada.

        Guarda el historial actual antes de borrar si existe. Crea un nuevo
        AlgoritmoGenetico y restablece todos los contadores y estados visuales.
        """
        if self.historial:
            self.guardar()
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.ga = AlgoritmoGenetico()
        self.historial, self.record, self.racha = [], 0, 0
        self.finished, self.phase, self.cola, self.espera = False, "idle", [], 0.0
        self._cache_stats = None
        self.tabla.delete(*self.tabla.get_children())
        self._nueva_escena()
        self._set_flow(None)
        self._set_status("idle", "Listo")
        self._actualizar_botones()
        self._actualizar_cromosoma()
        self._dibujar_grafica()
        self._actualizar_stats()
        self._pintar()

    def _agendar(self):
        self.after_id = self.root.after(FRAME_MS, self._tick)

    def _pausa(self):
        """Segundos que se muestra cada paso del flujo; se acorta con la velocidad."""
        return 0.08 if self.velocidad > 8 else 0.5 / math.sqrt(self.velocidad)

    def _tick(self):
        self.after_id = None
        if not self.running:
            return
        try:
            self._avanzar()
        finally:
            if self.running and self.after_id is None:
                self._agendar()

    def _avanzar(self):
        ahora = time.perf_counter()
        dt = min(ahora - self.t_prev, 0.1)
        self.t_prev = ahora
        if self.espera > 0:
            self.espera -= dt
        elif self.phase == "run":
            sim = self.sim
            self.acum += dt * 60 * self.velocidad      # 1x = 60 fotogramas por segundo reales
            pasos, self.acum = int(self.acum), self.acum % 1
            for _ in range(pasos):
                sim.paso()
                if sim.terminada:
                    break
            if sim.terminada:
                self._fin_evaluacion()
            else:
                self._pintar()
        elif self.phase == "cola":
            self._ejecutar(self.cola.pop(0))
        self._actualizar_stats()

    def _actualizar_tabla(self):
        """Top de intentos (generaciones): mayor score primero; a igual score, mayor promedio."""
        self.tabla.delete(*self.tabla.get_children())
        top = sorted(self.historial, key=lambda e: (-e["mejor"], -e["prom"], e["gen"]))[:TOP_INTENTOS]
        ultimo = self.historial[-1]["gen"] if self.historial else None
        for puesto, e in enumerate(top, 1):
            tags = (("meta",) if e["mejor"] >= META_SCORE else ()) + (("actual",) if e["gen"] == ultimo else ())
            self.tabla.insert("", "end", values=(puesto, e["gen"], e["mejor"], f"{e['prom']:.0f}"), tags=tags)

    # ── flujo del algoritmo ─────────────────────────────────────────────────
    def _fin_evaluacion(self):
        sim = self.sim
        aptitudes = sim.aptitudes()
        self.ga.evaluar(aptitudes)
        mejor, prom = max(aptitudes), sum(aptitudes) / len(aptitudes)
        self.record = max(self.record, mejor)
        self.racha = self.racha + 1 if mejor >= META_SCORE else 0
        self.historial.append({"gen": self.ga.generacion, "mejor": mejor, "prom": round(prom, 1),
                               "cromosoma": [round(g, 4) for g in self.ga.poblacion[0]]})
        self._actualizar_tabla()
        self._dibujar_grafica()
        self._actualizar_cromosoma()

        self.dead_shown = sim.lider
        logro = mejor >= META_SCORE
        self._pintar("¡ M E T A !" if logro else "G A M E   O V E R",
                     f"Generación {self.ga.generacion}  ·  mejor {mejor}  ·  promedio {prom:.0f}")
        self._set_flow("evaluar", f"Aptitud evaluada: mejor {mejor}")
        self.phase = "cola"
        self.cola = ["cruzar", "mutar", "actualizar", "parada", "poblacion", "evaluar"]
        self.espera = self._pausa() * 2

    def _ejecutar(self, paso):
        if paso == "cruzar":
            self.hijos = self.ga.cruzar()
            self._set_flow("cruzar", f"Cruce de un punto · prob. {self.ga.p_cruce:.2f}")
        elif paso == "mutar":
            self.ga.mutar(self.hijos)
            self._set_flow("mutar", f"Mutación de un gen · prob. {self.ga.p_mutacion:.2f}")
        elif paso == "actualizar":
            self.ga.actualizar(self.hijos)
            self._set_flow("actualizar", f"{ELITE} élites + {len(self.hijos)} hijos")
        elif paso == "parada":
            self._set_flow("parada", "Evaluando criterio de parada")
            if self.racha >= RACHA_CONVERGENCIA:
                return self._terminar(f"Convergió en la generación {self.ga.generacion - 1}", "done")
            if self.ga.generacion > self.max_gen:
                return self._terminar("Máximo de generaciones", "pause")
        elif paso == "poblacion":
            self._set_flow("poblacion", f"Generación {self.ga.generacion}")
            self._nueva_escena()
            self._pintar()
        elif paso == "evaluar":
            self._set_flow("evaluar", "Corriendo la simulación")
            self.phase = "run"
            self.acum = 0.0
            return
        self.espera = self._pausa()

    def _terminar(self, motivo, tipo):
        self.running = False
        self.finished = True
        self.phase = "cola"
        self.cola = ["poblacion", "evaluar"]
        self._set_status(tipo, motivo)
        self._set_flow("parada", motivo)
        self._actualizar_botones()
        self.guardar()

    # ── persistencia ────────────────────────────────────────────────────────
    def guardar(self):
        """Persiste los resultados de la sesión actual en resultados_dino_ga.json.

        Guarda un objeto JSON con:
          - Timestamp de guardado.
          - Parámetros del AG usados en esta sesión.
          - Nombre de los genes del cromosoma.
          - Número de generaciones evaluadas y si convergió.
          - Mejor score alcanzado y su cromosoma correspondiente.
          - Historial completo: por cada generación, el mejor score, el
            promedio y el cromosoma líder.

        No hace nada si no hay historial (el AG no llegó a evaluar ninguna
        generación). Los errores de escritura se ignoran silenciosamente para
        no interrumpir la interfaz gráfica.
        """
        if not self.historial:
            return
        datos = {
            "guardado": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parametros": {"poblacion": POBLACION, "elite": ELITE, "prob_cruce": self.ga.p_cruce,
                           "prob_mutacion": self.ga.p_mutacion, "max_generaciones": self.max_gen,
                           "meta_score": META_SCORE},
            "genes": [n for n, _, _ in GENES],
            "generaciones_evaluadas": len(self.historial),
            "convergio": self.racha >= RACHA_CONVERGENCIA,
            "mejor_score": self.record,
            "mejor_cromosoma": dict(zip([n for n, _, _ in GENES], [round(g, 4) for g in self.ga.mejor[1]])),
            "historial": self.historial,
        }
        try:
            with open(RESULT_PATH, "w", encoding="utf-8") as f:
                json.dump(datos, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _cerrar(self):
        self.guardar()
        self.root.destroy()


def main():
    """Punto de entrada principal de la aplicación.

    Configura DPI awareness en Windows para que la GUI no se vea borrosa en
    pantallas de alta resolución. Usa un socket de escucha en el puerto 47653
    para detectar si ya hay una instancia corriendo y evitar abrirla dos veces.
    Crea la ventana raíz de Tkinter y lanza el loop principal de eventos.
    """
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    # una sola instancia: si ya hay una ventana abierta, esta no abre otra
    cerrojo = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        cerrojo.bind(("127.0.0.1", 47653))
    except OSError:
        print("El programa ya está abierto en otra ventana.")
        return
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
