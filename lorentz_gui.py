#!/usr/bin/env python3
"""
Interfaz grafica -- Modelo de Lorentz con multiples osciladores
=================================================================
Permite agregar/quitar cualquier numero de osciladores (resonancias),
cada uno con su propia posicion (lambda_j), amortiguamiento (gamma_j)
y fuerza de oscilador (f_j), y grafica n(lambda) y alpha(lambda)
resultantes de sumar sobre todos ellos.
"""

import json
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# -----------------------Constantes fisicas ------------------------------------
Q    = 1.602176634e-19
M    = 9.1093837015e-31
EPS0 = 8.8541878128e-12
C    = 2.99792458e8


def w_de_nm(lamda_nm):
    """longitud de onda [nm] -> frecuencia angular [rad/s]"""
    return 2 * np.pi * C / (np.asarray(lamda_nm, dtype=float) * 1e-9)


def w_de_cm1(numero_onda_cm1):
    """numero de onda [cm^-1] -> frecuencia angular [rad/s]"""
    return 2 * np.pi * C * (numero_onda_cm1 * 100)


def nm_de_cm1(numero_onda_cm1):
    """numero de onda [cm^-1] -> longitud de onda [nm] (atajo util para IR)"""
    return 1e7 / numero_onda_cm1


# Fisica: calculo de n(lambda) y alpha(lambda) para j osciladores
def calcular_n_alpha(lamda_array_nm, osciladores, N, exacto=True):
    """
    osciladores: lista de tuplas (lamda_j_nm, gamma_j, f_j)
    N: densidad de osciladores [1/m^3]
    exacto: si True usa n=Re[sqrt(1+chi_e)] (recomendado, nunca da n<0
            espureo); si False usa la aproximacion n=1+chi_e/2 (mas
            simple pero puede fallar cerca de resonancias fuertes)
    """
    w = w_de_nm(lamda_array_nm)

    if exacto:
        chi_e = np.zeros_like(w, dtype=complex)
        for lamda_j, gamma_j, f_j in osciladores:
            w_j = w_de_nm(lamda_j)
            denom = w_j**2 - w**2 - 1j * gamma_j * w
            chi_e += (N * Q**2) / (M * EPS0) * f_j / denom
        n_complejo = np.sqrt(1 + chi_e)
        n = n_complejo.real
        kappa = n_complejo.imag
        alpha = 2 * kappa * w / C
    else:
        n = np.ones_like(w)
        alpha = np.zeros_like(w)
        for lamda_j, gamma_j, f_j in osciladores:
            w_j = w_de_nm(lamda_j)
            denom = (w_j**2 - w**2)**2 + gamma_j**2 * w**2
            n += (N * Q**2) / (2 * M * EPS0) * f_j * (w_j**2 - w**2) / denom
            alpha += (N * Q**2 * w**2) / (M * EPS0 * C) * f_j * gamma_j / denom

    return n, alpha

# Ventana emergente para agregar / editar un oscilador
class DialogoOscilador(tk.Toplevel):
    def __init__(self, parent, valores_iniciales=None):
        super().__init__(parent)
        self.title("Oscilador")
        self.resizable(False, False)
        self.resultado = None
        self.grab_set()

        # ---- modo de entrada de la posicion de resonancia ----
        ttk.Label(self, text="Posicion de la resonancia:").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(10, 2))

        self.modo = tk.StringVar(value="nm")
        fila_modo = ttk.Frame(self)
        fila_modo.grid(row=1, column=0, columnspan=3, sticky="w", padx=8)
        ttk.Radiobutton(fila_modo, text="nm", variable=self.modo, value="nm",
                         command=self._actualizar_etiqueta).pack(side="left")
        ttk.Radiobutton(fila_modo, text="cm⁻¹ (IR)", variable=self.modo, value="cm1",
                         command=self._actualizar_etiqueta).pack(side="left")

        self.etiqueta_pos = ttk.Label(self, text="lambda_j (nm):")
        self.etiqueta_pos.grid(row=2, column=0, sticky="e", padx=8, pady=4)
        self.entry_pos = ttk.Entry(self, width=15)
        self.entry_pos.grid(row=2, column=1, padx=4, pady=4)

        ttk.Label(self, text="gamma_j (rad/s):").grid(row=3, column=0, sticky="e", padx=8, pady=4)
        self.entry_gamma = ttk.Entry(self, width=15)
        self.entry_gamma.grid(row=3, column=1, padx=4, pady=4)

        ttk.Label(self, text="f_j (fuerza osc.):").grid(row=4, column=0, sticky="e", padx=8, pady=4)
        self.entry_f = ttk.Entry(self, width=15)
        self.entry_f.grid(row=4, column=1, padx=4, pady=4)

        if valores_iniciales:
            lamda_nm, gamma_j, f_j = valores_iniciales
            self.entry_pos.insert(0, str(lamda_nm))
            self.entry_gamma.insert(0, str(gamma_j))
            self.entry_f.insert(0, str(f_j))

        botones = ttk.Frame(self)
        botones.grid(row=5, column=0, columnspan=3, pady=10)
        ttk.Button(botones, text="Aceptar", command=self._aceptar).pack(side="left", padx=5)
        ttk.Button(botones, text="Cancelar", command=self.destroy).pack(side="left", padx=5)

        self.bind("<Return>", lambda e: self._aceptar())

    def _actualizar_etiqueta(self):
        if self.modo.get() == "nm":
            self.etiqueta_pos.config(text="lambda_j (nm):")
        else:
            self.etiqueta_pos.config(text="numero de onda (cm⁻¹):")

    def _aceptar(self):
        try:
            pos = float(self.entry_pos.get())
            gamma_j = float(self.entry_gamma.get())
            f_j = float(self.entry_f.get())
        except ValueError:
            messagebox.showerror("Error", "Todos los campos deben ser numeros validos.")
            return

        if self.modo.get() == "cm1":
            lamda_nm = nm_de_cm1(pos)
        else:
            lamda_nm = pos

        if lamda_nm <= 0 or gamma_j <= 0:
            messagebox.showerror("Error", "lambda_j y gamma_j deben ser positivos.")
            return

        self.resultado = (lamda_nm, gamma_j, f_j)
        self.destroy()

# Aplicacion 
class AppLorentz(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Modelo de Lorentz -- multiples osciladores")
        self.geometry("1150x700")

        self.osciladores = []   # lista de (lamda_nm, gamma_j, f_j)

        self._construir_panel_izquierdo()
        self._construir_panel_derecho()

    # ------------------------------------------------------------
    def _construir_panel_izquierdo(self):
        panel = ttk.Frame(self, padding=10)
        panel.pack(side="left", fill="y")

        ttk.Label(panel, text="Osciladores (resonancias)", font=("", 11, "bold")).pack(anchor="w")

        columnas = ("lamda", "gamma", "f")
        self.tabla = ttk.Treeview(panel, columns=columnas, show="headings", height=10)
        self.tabla.heading("lamda", text="λⱼ (nm)")
        self.tabla.heading("gamma", text="γⱼ (rad/s)")
        self.tabla.heading("f", text="fⱼ")
        self.tabla.column("lamda", width=90, anchor="center")
        self.tabla.column("gamma", width=110, anchor="center")
        self.tabla.column("f", width=80, anchor="center")
        self.tabla.pack(fill="x", pady=5)

        fila_botones = ttk.Frame(panel)
        fila_botones.pack(fill="x", pady=(0, 10))
        ttk.Button(fila_botones, text="+ Agregar", command=self._agregar).pack(side="left", padx=2)
        ttk.Button(fila_botones, text="Editar", command=self._editar).pack(side="left", padx=2)
        ttk.Button(fila_botones, text="Eliminar", command=self._eliminar).pack(side="left", padx=2)

        fila_archivo = ttk.Frame(panel)
        fila_archivo.pack(fill="x", pady=(0, 15))
        ttk.Button(fila_archivo, text="Cargar config.", command=self._cargar_config).pack(side="left", padx=2)
        ttk.Button(fila_archivo, text="Guardar config.", command=self._guardar_config).pack(side="left", padx=2)

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=8)

        # ---- Parametros globales ----
        ttk.Label(panel, text="Parametros globales", font=("", 11, "bold")).pack(anchor="w")

        grid_params = ttk.Frame(panel)
        grid_params.pack(fill="x", pady=5)

        ttk.Label(grid_params, text="N (osciladores/m³):").grid(row=0, column=0, sticky="e", pady=3)
        self.entry_N = ttk.Entry(grid_params, width=14)
        self.entry_N.insert(0, "3.0e27")
        self.entry_N.grid(row=0, column=1, pady=3, padx=4)

        ttk.Label(grid_params, text="λ min (nm):").grid(row=1, column=0, sticky="e", pady=3)
        self.entry_lmin = ttk.Entry(grid_params, width=14)
        self.entry_lmin.insert(0, "100")
        self.entry_lmin.grid(row=1, column=1, pady=3, padx=4)

        ttk.Label(grid_params, text="λ max (nm):").grid(row=2, column=0, sticky="e", pady=3)
        self.entry_lmax = ttk.Entry(grid_params, width=14)
        self.entry_lmax.insert(0, "800")
        self.entry_lmax.grid(row=2, column=1, pady=3, padx=4)

        ttk.Label(grid_params, text="N° de puntos:").grid(row=3, column=0, sticky="e", pady=3)
        self.entry_npts = ttk.Entry(grid_params, width=14)
        self.entry_npts.insert(0, "3000")
        self.entry_npts.grid(row=3, column=1, pady=3, padx=4)

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=8)

        # ---- Opciones de calculo y escala ----
        ttk.Label(panel, text="Formula", font=("", 10, "bold")).pack(anchor="w")
        self.var_formula = tk.StringVar(value="exacta")
        ttk.Radiobutton(panel, text="Exacta  n=Re[√(1+χe)]  (recomendada)",
                         variable=self.var_formula, value="exacta").pack(anchor="w")
        ttk.Radiobutton(panel, text="Aproximada  n≈1+χe/2  (puede dar n<0 cerca de resonancias)",
                         variable=self.var_formula, value="aproximada").pack(anchor="w")

        ttk.Label(panel, text="Escala eje X", font=("", 10, "bold")).pack(anchor="w", pady=(8, 0))
        self.var_escala = tk.StringVar(value="lineal")
        ttk.Radiobutton(panel, text="Lineal", variable=self.var_escala, value="lineal").pack(anchor="w")
        ttk.Radiobutton(panel, text="Logaritmica", variable=self.var_escala, value="log").pack(anchor="w")

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(panel, text="Graficar", command=self._graficar).pack(fill="x", pady=3)
        ttk.Button(panel, text="Guardar figura (PNG)", command=self._guardar_figura).pack(fill="x", pady=3)

        # ---- ejemplo precargado ----
        self._cargar_ejemplo_sio2()

    # ------------------------------------------------------------
    def _construir_panel_derecho(self):
        panel = ttk.Frame(self)
        panel.pack(side="right", fill="both", expand=True)

        self.figura = Figure(figsize=(7, 6), dpi=100)
        self.ax1 = self.figura.add_subplot(211)
        self.ax2 = self.figura.add_subplot(212, sharex=self.ax1)
        self.ax1.set_ylabel("n (indice de refraccion)")
        self.ax2.set_ylabel("alpha (1/m)")
        self.ax2.set_xlabel("longitud de onda (nm)")
        self.figura.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figura, master=panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        barra = NavigationToolbar2Tk(self.canvas, panel)
        barra.update()


    # Manejo de la tabla de osciladores
    def _refrescar_tabla(self):
        self.tabla.delete(*self.tabla.get_children())
        for lamda_nm, gamma_j, f_j in self.osciladores:
            self.tabla.insert("", "end", values=(f"{lamda_nm:g}", f"{gamma_j:.3e}", f"{f_j:g}"))

    def _agregar(self):
        dlg = DialogoOscilador(self)
        self.wait_window(dlg)
        if dlg.resultado:
            self.osciladores.append(dlg.resultado)
            self._refrescar_tabla()

    def _editar(self):
        sel = self.tabla.selection()
        if not sel:
            messagebox.showinfo("Editar", "Selecciona un oscilador de la tabla primero.")
            return
        idx = self.tabla.index(sel[0])
        dlg = DialogoOscilador(self, valores_iniciales=self.osciladores[idx])
        self.wait_window(dlg)
        if dlg.resultado:
            self.osciladores[idx] = dlg.resultado
            self._refrescar_tabla()

    def _eliminar(self):
        sel = self.tabla.selection()
        if not sel:
            messagebox.showinfo("Eliminar", "Selecciona un oscilador de la tabla primero.")
            return
        idx = self.tabla.index(sel[0])
        del self.osciladores[idx]
        self._refrescar_tabla()

    def _cargar_ejemplo_sio2(self):
        """Precarga el ejemplo de SiO2 (UV + 3 fonones IR) visto en la conversacion."""
        self.osciladores = [
            (130.0, 1.0e14, 0.98644),
            (nm_de_cm1(1100.0), 4.0e13, 0.00740),
            (nm_de_cm1(800.0), 3.0e13, 0.00370),
            (nm_de_cm1(450.0), 2.0e13, 0.00247),
        ]
        self._refrescar_tabla()


    # Guardar / cargar configuracion (JSON) -- para reproducibilidad
    def _guardar_config(self):
        ruta = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("JSON", "*.json")])
        if not ruta:
            return
        datos = {
            "osciladores": self.osciladores,
            "N": self.entry_N.get(),
            "lmin": self.entry_lmin.get(),
            "lmax": self.entry_lmax.get(),
            "npts": self.entry_npts.get(),
            "formula": self.var_formula.get(),
            "escala": self.var_escala.get(),
        }
        with open(ruta, "w") as f:
            json.dump(datos, f, indent=2)
        messagebox.showinfo("Guardado", f"Configuracion guardada en:\n{ruta}")

    def _cargar_config(self):
        ruta = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not ruta:
            return
        with open(ruta) as f:
            datos = json.load(f)
        self.osciladores = [tuple(o) for o in datos["osciladores"]]
        self.entry_N.delete(0, "end"); self.entry_N.insert(0, datos.get("N", "3.0e27"))
        self.entry_lmin.delete(0, "end"); self.entry_lmin.insert(0, datos.get("lmin", "100"))
        self.entry_lmax.delete(0, "end"); self.entry_lmax.insert(0, datos.get("lmax", "800"))
        self.entry_npts.delete(0, "end"); self.entry_npts.insert(0, datos.get("npts", "3000"))
        self.var_formula.set(datos.get("formula", "exacta"))
        self.var_escala.set(datos.get("escala", "lineal"))
        self._refrescar_tabla()

    # Graficar
    def _graficar(self):
        if not self.osciladores:
            messagebox.showwarning("Sin osciladores", "Agrega al menos un oscilador antes de graficar.")
            return
        try:
            N = float(self.entry_N.get())
            lmin = float(self.entry_lmin.get())
            lmax = float(self.entry_lmax.get())
            npts = int(self.entry_npts.get())
        except ValueError:
            messagebox.showerror("Error", "Revisa que N, λ min, λ max y N° de puntos sean numeros validos.")
            return

        if lmin <= 0 or lmax <= lmin or npts < 10:
            messagebox.showerror("Error", "Rango de longitud de onda invalido.")
            return

        if self.var_escala.get() == "log":
            lamda = np.logspace(np.log10(lmin), np.log10(lmax), npts)
        else:
            lamda = np.linspace(lmin, lmax, npts)

        exacto = (self.var_formula.get() == "exacta")
        n, alpha = calcular_n_alpha(lamda, self.osciladores, N, exacto=exacto)

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(lamda, n, color="tab:blue")
        self.ax2.plot(lamda, alpha, color="tab:red")

        for lamda_j, _, _ in self.osciladores:
            if lmin <= lamda_j <= lmax:
                self.ax1.axvline(lamda_j, color="gray", linestyle="--", alpha=0.5)
                self.ax2.axvline(lamda_j, color="gray", linestyle="--", alpha=0.5)

        if self.var_escala.get() == "log":
            self.ax1.set_xscale("log")
            self.ax2.set_xscale("log")

        self.ax1.set_ylabel("n (indice de refraccion)")
        self.ax2.set_ylabel("alpha (1/m)")
        self.ax2.set_xlabel("longitud de onda (nm)")
        self.ax1.grid(alpha=0.3, which="both")
        self.ax2.grid(alpha=0.3, which="both")

        self.figura.tight_layout()
        self.canvas.draw()

    def _guardar_figura(self):
        ruta = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png")])
        if ruta:
            self.figura.savefig(ruta, dpi=150)
            messagebox.showinfo("Guardado", f"Figura guardada en:\n{ruta}")


if __name__ == "__main__":
    app = AppLorentz()
    app.mainloop()
