"""Aterrizaje de nave + Algoritmo Genético.

Una flota de naves aprende a aterrizar suavemente sobre una plataforma sin
dañarse. Sigue el flujo visto en clase:

    Definir población -> Evaluar (función objetivo) -> Cruzar -> Mutar
    -> Actualizar población -> ¿Criterio de parada? -> (repetir)

Cromosoma (5 genes): son las "ganancias" del piloto automático de la nave.
    F1  ganancia horizontal: qué tan rápido intenta centrarse sobre la plataforma
    F2  ganancia de inclinación: cuánto se inclina para corregir su velocidad lateral
    F3  amortiguación de giro: frena las oscilaciones al inclinarse
    F4  velocidad de contacto: velocidad de descenso que busca al tocar el suelo
    F5  frenado por altura: cuánto más rápido baja cuando está alto

Función objetivo: puntaje de 0 a 1000. Un aterrizaje seguro (suave, recto y sobre
la plataforma) vale 800 o más; el resto se puntúa por lo cerca que estuvo.
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
POBLACION = 20            # naves por generación
ELITE = 2                 # las mejores pasan intactas a la siguiente generación
TORNEO = 3                # tamaño del torneo para escoger padres
PROB_CRUCE = 0.80
PROB_MUTACION = 0.75
TOP_INTENTOS = 10         # filas de la tabla de mejores intentos
MAX_GENERACIONES = 100    # criterio de parada 1: número de iteraciones
TASA_META = 0.5           # criterio de parada 2: valor convergente ...
RACHA_CONVERGENCIA = 2    # ... la mitad de la flota aterriza bien, N generaciones seguidas

# Genes: (nombre, mínimo, máximo)
GENES = [
    ("Ganancia horizontal", 0.0, 10.0),
    ("Ganancia de inclinación", 0.0, 5.0),
    ("Amortiguación de giro", 0.0, 10.0),
    ("Velocidad de contacto", 0.2, 3.0),
    ("Frenado por altura", 0.0, 10.0),
]

# ── Física (unidades: píxeles y fotogramas) ─────────────────────────────────
W, H = 880, 300
GROUND = 272
PAD_W = 130                 # ancho de la plataforma
START_ALT = 205
GRAV = 0.02
A_MAX = 0.055               # aceleración máxima del motor
ALPHA_MAX = 0.0025          # aceleración angular máxima
KP_GIRO = 0.012
KV = 0.12                   # ganancia vertical del piloto automático
FUEL_RATE = 0.25            # combustible gastado por fotograma a plena potencia
FUEL0 = 100.0
MAX_FRAMES = 1500
FOOT = (14.0, -15.0)        # posición de las patas respecto al centro
VY_SEGURA, VX_SEGURA, ANG_SEGURA = 1.2, 0.8, 0.20   # tolerancias de aterrizaje
SCORE_OK = 800              # score mínimo de un aterrizaje seguro

# ── Estilo ──────────────────────────────────────────────────────────────────
FONT, MONO = "Segoe UI", "Consolas"
BG, CARD, LINE, LINE2 = "#f4f4f2", "#ffffff", "#e7e5e4", "#d6d3d1"
TEXT, MUTED, FAINT, DARK = "#1c1917", "#78716c", "#a8a29e", "#1c1917"
INK, GHOST, GREEN, BLUE = "#535353", "#bdbdbd", "#16a34a", "#2563eb"
AMBER, RED = "#f59e0b", "#dc2626"
FRAME_MS = 16

SPEEDS = [("1x", 1), ("2x", 2), ("4x", 4), ("8x", 8), ("Turbo", 40)]
RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_nave_ga.json")


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


# ════════════════════════════════════════════════════════════════════════════
#  ALGORITMO GENÉTICO
# ════════════════════════════════════════════════════════════════════════════
class AlgoritmoGenetico:
    def __init__(self):
        self.p_cruce = PROB_CRUCE
        self.p_mutacion = PROB_MUTACION
        self.generacion = 1
        self.poblacion = []      # cromosomas; tras evaluar quedan ordenados de mejor a peor
        self.aptitud = []
        self.mejor = None        # (aptitud, cromosoma) récord histórico
        self.definir_poblacion()

    # 1. Definir población: cromosomas aleatorios dentro del rango de cada gen
    def definir_poblacion(self):
        self.poblacion = [[random.uniform(lo, hi) for _, lo, hi in GENES] for _ in range(POBLACION)]
        self.aptitud = []

    # 2. Evaluar: la aptitud la calcula la simulación (función objetivo)
    def evaluar(self, aptitudes):
        ranking = sorted(zip(aptitudes, self.poblacion), key=lambda p: p[0], reverse=True)
        self.aptitud = [a for a, _ in ranking]
        self.poblacion = [g for _, g in ranking]
        if self.mejor is None or self.aptitud[0] >= self.mejor[0]:
            self.mejor = (self.aptitud[0], self.poblacion[0][:])

    def _torneo(self):
        candidatos = random.sample(range(len(self.poblacion)), TORNEO)
        return self.poblacion[min(candidatos)]   # el índice menor es el mejor

    # 3. Cruzar: cruce de un punto con probabilidad p_cruce
    def cruzar(self):
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
        for hijo in hijos:
            if random.random() < self.p_mutacion:
                i = random.randrange(len(GENES))
                _, lo, hi = GENES[i]
                hijo[i] = clamp(hijo[i] + random.gauss(0, (hi - lo) * 0.15), lo, hi)

    # 5. Actualizar población: élite + hijos
    def actualizar(self, hijos):
        elite = [g[:] for g in self.poblacion[:ELITE]]
        self.poblacion = elite + hijos
        self.aptitud = []
        self.generacion += 1


# ════════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN DEL ATERRIZAJE (la función objetivo)
# ════════════════════════════════════════════════════════════════════════════
class Nave:
    __slots__ = ("genes", "x", "y", "vx", "vy", "th", "om", "fuel", "thr", "estado", "score", "integridad")

    def __init__(self, genes, x, vx, th):
        self.genes = genes
        self.x, self.y = x, START_ALT
        self.vx, self.vy = vx, 0.0
        self.th, self.om = th, 0.0
        self.fuel = FUEL0
        self.thr = 0.0
        self.estado = "vuelo"        # vuelo | aterrizado | estrellado | perdido
        self.score = 0
        self.integridad = 100


class Simulacion:
    """Todas las naves intentan aterrizar en el mismo escenario a la vez."""

    def __init__(self, cromosomas, rng=None):
        rng = rng or random.Random()
        self.pad_x = rng.uniform(300, 580)
        self.frame = 0
        self.terminada = False
        self.naves = []
        for g in cromosomas:
            x0 = self.pad_x + rng.choice((-1, 1)) * rng.uniform(90, 300)
            self.naves.append(Nave(g, clamp(x0, 60, W - 60), 0.0, 0.0))
        # todas parten del mismo punto y con la misma perturbación inicial
        x0, vx0, th0 = self.naves[0].x, rng.uniform(-1.2, 1.2), rng.uniform(-0.25, 0.25)
        for n in self.naves:
            n.x, n.vx, n.th = x0, vx0, th0
        self.en_vuelo = len(self.naves)
        self.lider = self.naves[0]

    def _cerrar(self, n, estado):
        n.estado = estado
        self.en_vuelo -= 1
        n.score = self._puntuar(n)

    def _puntuar(self, n):
        r = max(abs(n.vy) / VY_SEGURA, abs(n.vx) / VX_SEGURA, abs(n.th) / ANG_SEGURA)
        if n.estado == "aterrizado":
            return 800 + round(100 * (1 - min(r, 1))) + round(100 * n.fuel / FUEL0)
        dx = abs(n.x - self.pad_x)
        cerca = max(0.0, 1 - dx / 400)
        suave = max(0.0, 1 - max(0.0, abs(n.vy) - VY_SEGURA) / 4)
        recta = max(0.0, 1 - max(0.0, abs(n.th) - ANG_SEGURA) / 1.0)
        return round(350 * cerca + 200 * suave + 150 * recta)

    def paso(self):
        self.frame += 1
        for n in self.naves:
            if n.estado != "vuelo":
                continue
            f1, f2, f3, f4, f5 = n.genes
            # ── piloto automático (decide con los genes) ──
            vx_obj = clamp(f1 * 0.003 * (self.pad_x - n.x), -2.5, 2.5)
            th_obj = clamp(f2 * (vx_obj - n.vx), -0.7, 0.7)
            alpha = clamp(KP_GIRO * (th_obj - n.th) - f3 * 0.06 * n.om, -ALPHA_MAX, ALPHA_MAX)
            altura = max(0.0, n.y + FOOT[1])
            vy_obj = -(f4 + f5 * 0.003 * altura)
            thr = clamp((GRAV + KV * (vy_obj - n.vy)) / A_MAX, 0.0, 1.0) if n.fuel > 0 else 0.0
            n.thr = thr
            n.fuel = max(0.0, n.fuel - thr * FUEL_RATE)
            # ── física ──
            s, c = math.sin(n.th), math.cos(n.th)
            n.vx += thr * A_MAX * s
            n.vy += thr * A_MAX * c - GRAV
            n.x += n.vx
            n.y += n.vy
            n.om += alpha
            n.th += n.om

            if n.x < 0 or n.x > W or n.th > 1.6 or n.th < -1.6:
                self._cerrar(n, "perdido")
                continue
            s, c = math.sin(n.th), math.cos(n.th)
            pie_bajo = n.y + min(-FOOT[0] * s + FOOT[1] * c, FOOT[0] * s + FOOT[1] * c)
            if pie_bajo <= 0:
                n.y -= pie_bajo
                seguro = (abs(n.vy) <= VY_SEGURA and abs(n.vx) <= VX_SEGURA and abs(n.th) <= ANG_SEGURA
                          and abs(n.x - self.pad_x) <= PAD_W / 2 - 10)
                r = max(abs(n.vy) / VY_SEGURA, abs(n.vx) / VX_SEGURA, abs(n.th) / ANG_SEGURA)
                n.integridad = 100 if r <= 1 else max(0, round(100 * (1 - (r - 1) / 1.5)))
                self._cerrar(n, "aterrizado" if seguro else "estrellado")
                if seguro:
                    n.vx = n.vy = n.om = n.thr = 0.0

        if self.en_vuelo == 0 or self.frame >= MAX_FRAMES:
            for n in self.naves:
                if n.estado == "vuelo":
                    self._cerrar(n, "perdido")
            self.terminada = True
            self.lider = max(self.naves, key=lambda n: n.score)

    def aptitudes(self):
        return [n.score for n in self.naves]

    def exitos(self):
        return sum(1 for n in self.naves if n.estado == "aterrizado")


# ════════════════════════════════════════════════════════════════════════════
#  WIDGETS
# ════════════════════════════════════════════════════════════════════════════
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


# Geometría de la nave (x a la derecha, y hacia arriba, centro en el origen)
CASCO = [(-9, -8), (9, -8), (10, 4), (5, 12), (-5, 12), (-10, 4)]
TOBERA = [(-4, -8), (4, -8), (6, -12), (-6, -12)]
PATAS = [[(-7, -8), (-14, -15), (-18, -15)], [(7, -8), (14, -15), (18, -15)]]


def girar(puntos, x, y, th):
    """Rota los puntos th radianes (horario) y los pasa a coordenadas de pantalla."""
    s, c = math.sin(th), math.cos(th)
    salida = []
    for px, py in puntos:
        salida += [x + px * c + py * s, GROUND - (y - px * s + py * c)]
    return salida


# ════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN
# ════════════════════════════════════════════════════════════════════════════
class App:
    SIDE_W = 300
    CHART_W, CHART_H = 556, 214

    def __init__(self, root):
        self.root = root
        root.title("Nave · Algoritmo Genético")
        root.configure(bg=BG)
        root.resizable(False, False)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        ww, wh = 1252, min(830, sh - 70)
        root.geometry(f"{ww}x{wh}+{max(0, (sw - ww) // 2)}+{max(0, (sh - wh) // 2 - 20)}")

        self.ga = AlgoritmoGenetico()
        self.sim = None
        self.historial = []
        self.record = 0
        self.racha = 0
        self.tasa = 0.0
        self.max_gen = MAX_GENERACIONES
        self.velocidad = 4
        self.running = False
        self.finished = False
        self.phase = "idle"
        self.cola = []
        self.espera = 0
        self.hijos = []
        self.after_id = None
        self._cache_stats = None

        rng = random.Random(11)
        self.estrellas = [(rng.uniform(0, W), rng.uniform(10, GROUND - 90), rng.choice((1, 1, 2))) for _ in range(46)]
        self.crateres = [(rng.uniform(0, W), rng.randint(8, 22), rng.randint(3, 10)) for _ in range(46)]

        self._estilo_tabla()
        self._construir()
        self._nueva_escena()
        self._set_flow(None)
        self._set_status("idle", "Listo")
        self._actualizar_cromosoma()
        self._dibujar_grafica()
        self._pintar()
        root.protocol("WM_DELETE_WINDOW", self._cerrar)

    def _estilo_tabla(self):
        st = ttk.Style()
        st.theme_use("clam")
        st.configure("Nave.Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT,
                     borderwidth=0, rowheight=25, font=(FONT, -12))
        st.configure("Nave.Treeview.Heading", background=CARD, foreground=MUTED, borderwidth=0,
                     relief="flat", font=(FONT, -11, "bold"))
        st.map("Nave.Treeview.Heading", background=[("active", CARD)])
        st.map("Nave.Treeview", background=[("selected", "#f0efed")], foreground=[("selected", TEXT)])
        st.layout("Nave.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])

    # ── layout ──────────────────────────────────────────────────────────────
    def _construir(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=24, pady=20)

        head = tk.Frame(outer, bg=BG)
        head.pack(fill="x", pady=(0, 16))
        izq = tk.Frame(head, bg=BG)
        izq.pack(side="left")
        etiqueta(izq, "Nave  ·  Algoritmo Genético", -22, "bold", bg=BG).pack(anchor="w")
        etiqueta(izq, f"Flota de {POBLACION} naves  ·  cromosoma de {len(GENES)} genes  ·  meta: "
                      f"{int(TASA_META * 100)} % de la flota aterriza sin daño",
                 -12, fg=MUTED, bg=BG).pack(anchor="w", pady=(2, 0))
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
        c = self.canvas
        for x, y, r in self.estrellas:
            c.create_oval(x, y, x + r, y + r, fill=LINE2, outline="")
        c.create_line(0, GROUND, W, GROUND, fill=INK, width=2)
        for x, off, ln in self.crateres:
            c.create_line(x, GROUND + off, x + ln, GROUND + off, fill=FAINT, width=2)

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
        self.tabla = ttk.Treeview(hist, columns=("puesto", "gen", "mejor", "exitos"), show="headings",
                                  style="Nave.Treeview", selectmode="none", height=TOP_INTENTOS)
        for col, texto, ancho in (("puesto", "#", 34), ("gen", "Intento", 62), ("mejor", "Score", 66),
                                  ("exitos", "Aterrizajes", 84)):
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
        celdas = [("gen", "GENERACIÓN"), ("vuelo", "EN VUELO"), ("record", "RÉCORD"), ("conv", "CONVERGENCIA")]
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
        self.tasa_txt = etiqueta(interior, "", -11, fg=MUTED)
        self.tasa_txt.grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))

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
        titulo_seccion(interior, "MEJOR CROMOSOMA").pack(anchor="w", pady=(0, 2))
        self.genes_ui = []
        for i, (nombre, lo, hi) in enumerate(GENES):
            fila = tk.Frame(interior, bg=CARD)
            fila.pack(fill="x", pady=(5, 0))
            etiqueta(fila, f"F{i + 1}", -11, "bold", FAINT, fuente=MONO).pack(side="left")
            etiqueta(fila, nombre, -12, fg=MUTED).pack(side="left", padx=(8, 0))
            val = etiqueta(fila, "–", -12, "bold", fuente=MONO)
            val.pack(side="right")
            barra = tk.Canvas(interior, height=5, bg=CARD, highlightthickness=0)
            barra.pack(fill="x", pady=(2, 0))
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
            val.config(text=f"{g:.2f}")
            barra.delete("all")
            w = barra.winfo_width() if barra.winfo_width() > 1 else self.SIDE_W - 34
            barra.create_rectangle(0, 0, w, 5, fill=LINE, outline="")
            barra.create_rectangle(0, 0, w * (g - lo) / (hi - lo), 5, fill=DARK, outline="")

    def _dibujar_progreso(self):
        c = self.progreso
        c.delete("all")
        w = c.winfo_width()
        c.create_rectangle(0, 0, w, 6, fill=LINE, outline="")
        frac = clamp(self.tasa / TASA_META, 0, 1)
        c.create_rectangle(0, 0, w * frac, 6, fill=GREEN if frac >= 1 else DARK, outline="")

    def _actualizar_stats(self):
        if self.sim and self.phase == "run":
            vuelo = self.sim.en_vuelo
        else:
            vuelo = POBLACION if self.phase == "idle" else 0
        datos = (self.ga.generacion, vuelo, self.record, self.racha, round(self.tasa, 3))
        if datos == self._cache_stats:
            return
        self._cache_stats = datos
        self.stat["gen"].config(text=f"{datos[0]}")
        self.stat["vuelo"].config(text=f"{vuelo}/{POBLACION}")
        self.stat["record"].config(text=f"{datos[2]}")
        self.stat["conv"].config(text=f"{datos[3]}/{RACHA_CONVERGENCIA}")
        ok = round(self.tasa * POBLACION)
        self.tasa_txt.config(text=f"Aterrizajes seguros (última gen.): {ok}/{POBLACION}"
                                  f"  ·  meta {round(TASA_META * POBLACION)}")
        self._dibujar_progreso()

    def _dibujar_grafica(self):
        c = self.chart
        c.delete("all")
        w, h = self.CHART_W, self.CHART_H
        ml, mr, mt, mb = 44, 14, 12, 26
        pw, ph = w - ml - mr, h - mt - mb
        for v in range(0, 1001, 250):
            y = mt + ph - ph * v / 1000
            c.create_line(ml, y, w - mr, y, fill=LINE if v else LINE2)
            c.create_text(ml - 8, y, text=str(v), anchor="e", fill=FAINT, font=(FONT, -10))
        n = len(self.historial)
        xmax = max(10, n)
        paso = max(1, math.ceil(xmax / 10))

        def px(g):
            return ml + (g - 1) / (xmax - 1) * pw

        def py(v):
            return mt + ph - ph * min(v, 1000) / 1000

        for g in range(1, xmax + 1, paso):
            c.create_text(px(g), h - mb + 13, text=str(g), fill=FAINT, font=(FONT, -10))
        c.create_line(ml, py(SCORE_OK), w - mr, py(SCORE_OK), fill=GREEN, dash=(4, 3))
        c.create_text(w - mr, py(SCORE_OK) - 8, text="aterrizaje seguro", anchor="e", fill=GREEN,
                      font=(FONT, -10))
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

    def _actualizar_tabla(self):
        """Top de intentos (generaciones): mayor score primero; a igual score, más aterrizajes."""
        self.tabla.delete(*self.tabla.get_children())
        top = sorted(self.historial, key=lambda e: (-e["mejor"], -e["exitos"], e["gen"]))[:TOP_INTENTOS]
        ultimo = self.historial[-1]["gen"] if self.historial else None
        for puesto, e in enumerate(top, 1):
            tags = (("meta",) if e["mejor"] >= SCORE_OK else ()) + (("actual",) if e["gen"] == ultimo else ())
            self.tabla.insert("", "end", values=(puesto, e["gen"], e["mejor"], f"{e['exitos']}/{POBLACION}"),
                              tags=tags)

    # ── escena ──────────────────────────────────────────────────────────────
    def _nueva_escena(self):
        self.sim = Simulacion(self.ga.poblacion, random.Random(random.getrandbits(32)))

    def _dibujar_nave(self, c, n, lider):
        contorno, ancho = (INK, 2) if lider else (GHOST, 1.5)
        relleno = "#ffffff"
        if n.estado == "estrellado" or n.estado == "perdido":
            color = RED if lider else LINE2
            x = clamp(n.x, 10, W - 10)
            y = GROUND - max(0.0, n.y - 10) if n.estado == "estrellado" else GROUND - 8
            for dx, dy in ((-6, -6), (-6, 6)):
                c.create_line(x + dx, y + dy, x - dx, y - dy, fill=color, width=2, tags="dyn")
            if lider and n.estado == "estrellado":
                for dx, dy in ((-16, -2), (14, -6), (4, -13)):
                    c.create_line(x + dx, y + dy, x + dx * 1.3, y + dy * 1.3, fill=color, width=2, tags="dyn")
            return
        if lider and n.thr > 0.04:
            c.create_polygon(girar([(-4, -12), (4, -12), (0, -12 - 8 - 30 * n.thr)], n.x, n.y, n.th),
                             fill=AMBER, outline="", tags="dyn")
        for pata in PATAS:
            c.create_line(*girar(pata, n.x, n.y, n.th), fill=contorno, width=ancho, tags="dyn")
        c.create_polygon(girar(TOBERA, n.x, n.y, n.th), fill=relleno, outline=contorno, width=ancho, tags="dyn")
        c.create_polygon(girar(CASCO, n.x, n.y, n.th), fill=relleno, outline=contorno, width=ancho, tags="dyn")
        if lider:
            vx, vy = girar([(0, 4)], n.x, n.y, n.th)
            c.create_oval(vx - 3, vy - 3, vx + 3, vy + 3, fill=INK, outline="", tags="dyn")

    def _pintar(self, mensaje=None, sub=None, color=INK):
        c, sim = self.canvas, self.sim
        c.delete("dyn")

        # plataforma de aterrizaje
        px0, px1 = sim.pad_x - PAD_W / 2, sim.pad_x + PAD_W / 2
        hecho = any(n.estado == "aterrizado" for n in sim.naves)
        c.create_line(px0, GROUND - 2, px1, GROUND - 2, fill=GREEN if hecho else DARK, width=5, tags="dyn")
        for x in (px0 + 6, px1 - 6):
            c.create_line(x, GROUND - 4, x, GROUND - 12, fill=GREEN if hecho else FAINT, width=2, tags="dyn")

        # fantasmas primero, líder encima
        lider = sim.lider if sim.terminada else sim.naves[0]
        for n in sim.naves:
            if n is not lider:
                self._dibujar_nave(c, n, False)
        self._dibujar_nave(c, lider, True)

        vuelo = sim.en_vuelo if self.phase == "run" else 0 if self.phase != "idle" else POBLACION
        c.create_text(24, 26, anchor="w", fill=FAINT, font=(FONT, -12, "bold"), tags="dyn",
                      text=f"GENERACIÓN {self.ga.generacion}    ·    EN VUELO {vuelo}/{POBLACION}"
                           f"    ·    ATERRIZADAS {sim.exitos()}")
        alt = max(0.0, lider.y + FOOT[1])
        tele = (f"ALT {alt:03.0f}   VEL {abs(lider.vy):.1f}   INCL {math.degrees(lider.th):+03.0f}°"
                f"   COMB {lider.fuel:03.0f}%")
        if lider.estado != "vuelo":
            tele += f"   INTEG {lider.integridad}%"
        c.create_text(W - 24, 26, anchor="e", fill=INK, font=(MONO, -14, "bold"), tags="dyn", text=tele)
        if self.phase == "idle":
            c.create_text(W / 2, 110, fill=FAINT, font=(FONT, -14), tags="dyn",
                          text="Pulsa «Iniciar entrenamiento» para que la flota empiece a aprender a aterrizar")
        if mensaje:
            c.create_text(W / 2, 96, text=mensaje, fill=color, font=(MONO, -22, "bold"), tags="dyn")
        if sub:
            c.create_text(W / 2, 128, text=sub, fill=MUTED, font=(FONT, -13), tags="dyn")

    # ── control ─────────────────────────────────────────────────────────────
    def alternar(self):
        if self.running:
            self.detener()
        else:
            self.iniciar()

    def iniciar(self):
        if self.finished:
            self.finished = False
            self.racha = 0
        if self.ga.generacion > self.max_gen:
            self.max_gen = self.ga.generacion + 10
        if self.phase == "idle":
            self.cola = ["poblacion", "evaluar"]
            self.phase = "cola"
            self.espera = 0
        self.running = True
        self._set_status("run", "Entrenando")
        self._actualizar_botones()
        self._agendar()

    def detener(self):
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self._set_status("pause", "Detenido")
        self._actualizar_botones()
        self.guardar()

    def reiniciar(self):
        if self.historial:
            self.guardar()
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.ga = AlgoritmoGenetico()
        self.historial, self.record, self.racha, self.tasa = [], 0, 0, 0.0
        self.finished, self.phase, self.cola, self.espera = False, "idle", [], 0
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
        return max(3, 26 // min(self.velocidad, 8))

    def _tick(self):
        self.after_id = None
        if not self.running:
            return
        if self.espera > 0:
            self.espera -= 1
        elif self.phase == "run":
            sim = self.sim
            for _ in range(self.velocidad):
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
        if self.running:
            self._agendar()

    # ── flujo del algoritmo ─────────────────────────────────────────────────
    def _fin_evaluacion(self):
        sim = self.sim
        aptitudes = sim.aptitudes()
        exitos = sim.exitos()
        self.ga.evaluar(aptitudes)
        mejor, prom = max(aptitudes), sum(aptitudes) / len(aptitudes)
        self.record = max(self.record, mejor)
        self.tasa = exitos / POBLACION
        self.racha = self.racha + 1 if self.tasa >= TASA_META else 0
        self.historial.append({"gen": self.ga.generacion, "mejor": mejor, "prom": round(prom, 1),
                               "exitos": exitos, "cromosoma": [round(g, 4) for g in self.ga.poblacion[0]]})
        self._actualizar_tabla()
        self._dibujar_grafica()
        self._actualizar_cromosoma()

        detalle = f"Generación {self.ga.generacion}  ·  {exitos}/{POBLACION} naves aterrizaron sin daño  ·  mejor {mejor}"
        if sim.lider.estado == "aterrizado":
            self._pintar("A T E R R I Z A J E   E X I T O S O",
                         detalle + f"  ·  integridad {sim.lider.integridad}%", GREEN)
        else:
            self._pintar("N A V E   D A Ñ A D A", detalle, INK)
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
        elif paso == "evaluar":
            self._set_flow("evaluar", "Corriendo la simulación")
            self._nueva_escena()
            self.phase = "run"
            self._pintar()
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
        if not self.historial:
            return
        datos = {
            "guardado": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parametros": {"poblacion": POBLACION, "elite": ELITE, "prob_cruce": self.ga.p_cruce,
                           "prob_mutacion": self.ga.p_mutacion, "max_generaciones": self.max_gen,
                           "meta_tasa_aterrizajes": TASA_META, "racha_convergencia": RACHA_CONVERGENCIA},
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
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    # una sola instancia: si ya hay una ventana abierta, esta no abre otra
    cerrojo = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        cerrojo.bind(("127.0.0.1", 47654))
    except OSError:
        print("El programa ya está abierto en otra ventana.")
        return
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
