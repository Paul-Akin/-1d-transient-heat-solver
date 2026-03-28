"""
Multilayer Transient Heat Conduction Solver -- GUI Edition
==========================================================
Wraps the original Temp3.py solver with a Tkinter interface.

Physics:
  Explicit finite-difference (FTCS) scheme for the 1-D heat equation:
      rho*cp * dT/dt = d/dz[ k(z) * dT/dz ]
  Stability enforced via Fourier criterion: Fo = alpha*dt/dz^2 <= 0.5

Boundary conditions:
  Front surface: Neumann (constant heat flux q")
  Back surface:  Neumann (insulated, q = 0)

Interface conductivity: harmonic mean
  k_half = 2*k_i*k_{i+1} / (k_i + k_{i+1})

Units: U.S. Customary throughout.
"""

import sys
import io
import queue
import threading
import traceback as tb

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================
#  DATA CLASSES
# ============================================================

@dataclass
class PlateLayer:
    """Single material layer in the wall stack."""
    material_key: str
    thickness: float  # inches


@dataclass
class ThermalProperties:
    """
    Isotropic, temperature-independent thermal properties.

    Attributes
    ----------
    k   : Thermal conductivity,  BTU/(hr*ft*degF)
    rho : Density,               lb/ft^3
    cp  : Specific heat,         BTU/(lb*degF)
    """
    k:   float
    rho: float
    cp:  float

    @property
    def alpha(self) -> float:
        """Thermal diffusivity alpha = k/(rho*cp), ft^2/s."""
        return (self.k / 3600.0) / (self.rho * self.cp)


# ============================================================
#  MATERIAL DATABASE
# ============================================================

THERMAL_PROPERTIES: dict[str, ThermalProperties] = {
    'A572_GR50': ThermalProperties(
        k=34.9,   rho=490.0, cp=0.12   # Structural steel
    ),
    'C95900': ThermalProperties(
        k=41.1,   rho=440.6, cp=0.09   # Aluminum bronze
    ),
    'ASTM_B21_H02': ThermalProperties(
        k=35.0,   rho=535.0, cp=0.09   # Naval brass/bronze
    ),
}

MATERIAL_KEYS = list(THERMAL_PROPERTIES.keys())


# ============================================================
#  SOLVER  (logic preserved from Temp3.py)
# ============================================================

class MultilayerHeatConductionSolver:
    """
    Explicit FTCS finite-difference solver for multilayer transient conduction.

    Grid setup assigns material properties node-by-node based on cumulative
    layer thickness.  Interface conductivities use the harmonic mean to
    enforce flux continuity across material boundaries
    (see Patankar, 'Numerical Heat Transfer and Fluid Flow', 1980).
    """

    def __init__(self, layers: List[PlateLayer], thermal_props: dict):
        self.layers = layers
        self.thermal_props = thermal_props
        self.total_thickness = sum(lyr.thickness for lyr in layers)  # inches
        self._setup_grid()

    def _setup_grid(self, nodes_per_inch: int = 50):
        total_nodes = int(self.total_thickness * nodes_per_inch) + 1
        self.z   = np.linspace(0.0, self.total_thickness / 12.0, total_nodes)  # ft
        self.dz  = self.z[1] - self.z[0]

        self.k_array     = np.zeros(total_nodes)
        self.rho_array   = np.zeros(total_nodes)
        self.cp_array    = np.zeros(total_nodes)
        self.alpha_array = np.zeros(total_nodes)
        self.material_names: List[str] = []

        z_cum = 0.0
        for lyr in self.layers:
            z_end  = z_cum + lyr.thickness
            props  = self.thermal_props[lyr.material_key]
            mask   = (self.z * 12.0 >= z_cum) & (self.z * 12.0 <= z_end)
            self.k_array[mask]     = props.k
            self.rho_array[mask]   = props.rho
            self.cp_array[mask]    = props.cp
            self.alpha_array[mask] = props.alpha
            z_cum = z_end
            self.material_names.append(f"{lyr.material_key} ({lyr.thickness:.4f} in)")

        # Layer boundary positions (inches) for plotting
        self.layer_boundaries: List[float] = []
        z_cum = 0.0
        for lyr in self.layers:
            z_cum += lyr.thickness
            self.layer_boundaries.append(z_cum)

    def solve_transient(
        self,
        q_flux:         float,
        t_total:        float,
        T_initial:      float = 70.0,
        safety_factor:  float = 0.5,
        report_interval: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Integrate the heat equation forward in time.

        Parameters
        ----------
        q_flux          : Surface heat flux, BTU/(ft^2*s)
        t_total         : Total duration, s
        T_initial       : Uniform initial temperature, degF
        safety_factor   : dt = safety_factor * dt_max  (< 1 for stability)
        report_interval : Stdout progress interval, s

        Returns
        -------
        z_in      : Nodal positions, inches
        T_final   : Temperature distribution at t = t_total, degF
        T_history : Dict {time_key (float): T array} at ~73 snapshots
        """
        nz = len(self.z)

        # Stable time step from Von Neumann criterion: Fo = alpha*dt/dz^2 <= 0.5
        alpha_min = np.min(self.alpha_array[self.alpha_array > 0.0])
        dt_max    = self.dz ** 2 / (2.0 * alpha_min)
        dt        = safety_factor * dt_max
        nt        = int(np.ceil(t_total / dt))
        actual_dt = t_total / nt
        Fo        = alpha_min * actual_dt / self.dz ** 2

        print(f"\nMultilayer Heat Conduction Solver")
        print(f"{'=' * 68}")
        print(f"  Nodes              : {nz}")
        print(f"  Spatial step dz    : {self.dz * 12:.6f}  in")
        print(f"  Time step dt       : {actual_dt:.6f}  s")
        print(f"  Total time steps   : {nt}")
        print(f"  Fourier number Fo  : {Fo:.6f}")
        if Fo <= 0.5:
            print(f"  Stability check    : Fo = {Fo:.4f} <= 0.5  [STABLE]")
        else:
            print(f"  Stability WARNING  : Fo = {Fo:.4f} > 0.5   [CHECK RESULTS]")
        print()

        # Initial condition
        T = np.ones(nz) * T_initial

        # Snapshot schedule: ~73 evenly-spaced frames over [0, t_total]
        T_history: dict = {}
        save_times    = np.linspace(0.0, t_total, 73)
        save_indices  = {int(round(t / actual_dt)): float(t) for t in save_times}

        # Convert conductivity to BTU/(s*ft*degF) for flux BC
        k_s = self.k_array / 3600.0

        last_report = 0.0

        for n in range(nt):
            T_new = T.copy()

            # Interior: energy balance with harmonic-mean interface conductivities
            for i in range(1, nz - 1):
                k_p = 2.0 * k_s[i] * k_s[i+1] / (k_s[i] + k_s[i+1])
                k_m = 2.0 * k_s[i] * k_s[i-1] / (k_s[i] + k_s[i-1])
                q_p = k_p * (T[i+1] - T[i])   / self.dz
                q_m = k_m * (T[i]   - T[i-1])  / self.dz
                dT_dt    = (q_p - q_m) / (self.rho_array[i] * self.cp_array[i] * self.dz)
                T_new[i] = T[i] + dT_dt * actual_dt

            # Front BC: Neumann (prescribed heat flux q")
            T_new[0]  = T_new[1] + q_flux * self.dz / k_s[0]

            # Back BC: Neumann (insulated, dT/dz = 0)
            T_new[-1] = T_new[-2]

            T = T_new

            step_np1 = n + 1
            t_np1    = step_np1 * actual_dt

            if step_np1 in save_indices:
                T_history[save_indices[step_np1]] = T.copy()
            if n == nt - 1:
                T_history[float(t_total)] = T.copy()

            if report_interval:
                if (t_np1 - last_report >= report_interval) or (n == nt - 1):
                    pct = 100.0 * step_np1 / nt
                    print(f"  {pct:5.1f}%  |  t = {t_np1:7.3f} s  |  T_max = {T.max():7.1f} degF")
                    last_report = t_np1

        return self.z * 12.0, T, T_history  # z returned in inches

    def print_layer_temperatures(self, z_inches: np.ndarray,
                                  T_final: np.ndarray, T_initial: float):
        """Print temperature table at layer boundaries."""
        print("\n" + "=" * 72)
        print("LAYER BOUNDARY TEMPERATURES")
        print("=" * 72)
        print(f"{'Pos (in)':<12}  {'Boundary':<36}  {'T (degF)':<10}  dT (degF)")
        print("-" * 72)
        print(f"{0.0:<12.4f}  {'Front Surface':<36}  "
              f"{T_final[0]:>9.1f}   {T_final[0] - T_initial:>9.1f}")

        z_cum = 0.0
        for i, lyr in enumerate(self.layers):
            z_cum += lyr.thickness
            idx = np.argmin(np.abs(z_inches - z_cum))
            desc = (f"{self.layers[i].material_key} / {self.layers[i+1].material_key}"
                    if i < len(self.layers) - 1
                    else f"{self.layers[i].material_key} (Back Surface)")
            print(f"{z_cum:<12.4f}  {desc:<36}  "
                  f"{T_final[idx]:>9.1f}   {T_final[idx] - T_initial:>9.1f}")

        print("=" * 72 + "\n")


# ============================================================
#  GUI APPLICATION
# ============================================================

# Layer-band background colors (cycling)
_BAND_COLORS = ['#bbdefb', '#f8bbd0', '#c8e6c9', '#fff9c4', '#e1bee7']


class ThermalSolverGUI:
    """Tkinter front-end for MultilayerHeatConductionSolver."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Multilayer Transient Heat Conduction Solver")
        self.root.geometry("1380x840")
        self.root.minsize(1000, 650)

        # Internal state
        self._result_q:    queue.Queue = queue.Queue()
        self._cbar3:       Optional[object] = None
        self._layer_data:  List[dict] = []   # {'material': str, 'thickness': float}
        self._layer_vars:  List[tuple] = []  # [(mat_StringVar, thick_StringVar), ...]
        self._layer_frames: List[ttk.Frame] = []

        self._configure_styles()
        self._build_ui()
        self._load_defaults()

    # ----------------------------------------------------------
    #  Style
    # ----------------------------------------------------------

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

        style.configure('Run.TButton',
                        font=('Helvetica', 11, 'bold'),
                        foreground='white',
                        background='#2e7d32',
                        padding=6)
        style.map('Run.TButton',
                  background=[('active', '#1b5e20'), ('disabled', '#888')])

        style.configure('Del.TButton', foreground='#c62828', padding=2)
        style.configure('Add.TButton', foreground='#1565c0')

        style.configure('Header.TLabel',
                        font=('Helvetica', 12, 'bold'))
        style.configure('Sub.TLabel',
                        font=('Helvetica', 9, 'bold'))

    # ----------------------------------------------------------
    #  Top-level Layout
    # ----------------------------------------------------------

    def _build_ui(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left  = ttk.Frame(paned, width=375)
        right = ttk.Frame(paned)
        paned.add(left,  weight=0)
        paned.add(right, weight=1)

        self._build_controls(left)
        self._build_plot_area(right)

    # ----------------------------------------------------------
    #  Left Panel: Controls
    # ----------------------------------------------------------

    def _build_controls(self, parent: ttk.Frame):
        ttk.Label(parent, text="Heat Conduction Solver",
                  style='Header.TLabel').pack(pady=(10, 4))

        # ---------- Global parameters ----------
        gf = ttk.LabelFrame(parent, text="Load & Duration")
        gf.pack(fill=tk.X, padx=8, pady=4)

        self._q_var  = tk.StringVar(value="375.0")
        self._t_var  = tk.StringVar(value="7.2")
        self._T0_var = tk.StringVar(value="70.0")

        params = [
            ('Heat Flux q" [BTU/(ft\u00b2\u00b7s)]:', self._q_var),
            ('Duration t  [s]:',                      self._t_var),
            ('Initial Temp T\u2080  [\u00b0F]:',      self._T0_var),
        ]
        for label, var in params:
            row = ttk.Frame(gf)
            row.pack(fill=tk.X, padx=8, pady=3)
            ttk.Label(row, text=label, width=28, anchor='w').pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT)

        # ---------- Layer stack ----------
        lf = ttk.LabelFrame(parent,
                             text="Layer Stack  (heated surface  \u25b6  back surface)")
        lf.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Column headers
        hdr = ttk.Frame(lf)
        hdr.pack(fill=tk.X, padx=6, pady=(2, 0))
        for text, width in [('#', 3), ('Material', 15), ('Thick (in)', 10)]:
            ttk.Label(hdr, text=text, width=width,
                      anchor='center', style='Sub.TLabel').pack(side=tk.LEFT)
        ttk.Separator(lf, orient='horizontal').pack(fill=tk.X, padx=4, pady=2)

        # Scrollable layer list
        # Resolve the actual ttk frame background at runtime so the canvas
        # blends seamlessly instead of painting a white box over it.
        _ttk_bg = ttk.Style().lookup('TFrame', 'background')
        sc = tk.Canvas(lf, highlightthickness=0, bg=_ttk_bg)
        vsb = ttk.Scrollbar(lf, orient='vertical', command=sc.yview)
        sc.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner = ttk.Frame(sc)
        self._cwin  = sc.create_window((0, 0), window=self._inner, anchor='nw')
        self._inner.bind('<Configure>',
                         lambda e: sc.configure(scrollregion=sc.bbox('all')))
        sc.bind('<Configure>',
                lambda e: sc.itemconfig(self._cwin, width=e.width))
        self._scroll_canvas = sc

        # Scroll with mouse wheel
        sc.bind_all('<MouseWheel>',
                    lambda e: sc.yview_scroll(int(-1*(e.delta/120)), 'units'))

        # Add-layer button
        ttk.Button(lf, text="  +  Add Layer",
                   style='Add.TButton',
                   command=self._add_layer_clicked).pack(pady=(4, 2))

        # ---------- Run button ----------
        self._run_btn = ttk.Button(parent, text="  \u25b6   Run Analysis",
                                    command=self._on_run,
                                    style='Run.TButton')
        self._run_btn.pack(fill=tk.X, padx=8, pady=6)

        # ---------- Progress ----------
        self._prog = ttk.Progressbar(parent, mode='indeterminate', length=200)
        self._prog.pack(fill=tk.X, padx=8, pady=(0, 4))

        # ---------- Log ----------
        logf = ttk.LabelFrame(parent, text="Solver Log")
        logf.pack(fill=tk.BOTH, expand=False, padx=8, pady=4)
        self._log_widget = scrolledtext.ScrolledText(
            logf, height=11,
            font=('Courier New', 8),
            state='disabled',
            bg='#1e1e1e', fg='#d4d4d4',
            insertbackground='white')
        self._log_widget.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    # ----------------------------------------------------------
    #  Right Panel: Plot Tabs
    # ----------------------------------------------------------

    def _build_plot_area(self, parent: ttk.Frame):
        self._notebook = ttk.Notebook(parent)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        fig_specs = [
            ('Temperature Evolution', '_fig1', '_ax1', '_cv1'),
            ('Final \u0394T Profile',  '_fig2', '_ax2', '_cv2'),
            ('Contour Map',            '_fig3', '_ax3', '_cv3'),
        ]

        for title, fig_attr, ax_attr, cv_attr in fig_specs:
            fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
            setattr(self, fig_attr, fig)
            setattr(self, ax_attr,  ax)

            tab = ttk.Frame(self._notebook)
            self._notebook.add(tab, text=title)

            canvas = FigureCanvasTkAgg(fig, master=tab)
            toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas.draw()
            setattr(self, cv_attr, canvas)

        self._draw_placeholders()

    def _draw_placeholders(self):
        for ax, title in [
            (self._ax1, 'Temperature Evolution'),
            (self._ax2, 'Final \u0394T Profile'),
            (self._ax3, 'Contour Map'),
        ]:
            ax.set_facecolor('#f0f0f0')
            ax.text(0.5, 0.5, f'{title}\n(configure parameters and click Run)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='#aaaaaa', fontstyle='italic')
            ax.set_xticks([]); ax.set_yticks([])

        for cv in (self._cv1, self._cv2, self._cv3):
            cv.draw()

    # ----------------------------------------------------------
    #  Layer Stack Management
    # ----------------------------------------------------------

    def _load_defaults(self):
        defaults = [
            ('A572_GR50',    0.75),
            ('C95900',       0.093),
            ('ASTM_B21_H02', 0.53),
            ('C95900',       0.093),
        ]
        for mat, thick in defaults:
            self._layer_data.append({'material': mat, 'thickness': thick})
        self._rebuild_layer_ui()

    def _rebuild_layer_ui(self):
        """Destroy and recreate all layer row widgets from self._layer_data."""
        for f in self._layer_frames:
            f.destroy()
        self._layer_frames.clear()
        self._layer_vars.clear()

        for i, ldata in enumerate(self._layer_data):
            f = ttk.Frame(self._inner)
            f.pack(fill=tk.X, padx=2, pady=1)
            self._layer_frames.append(f)

            # Alternate row shading
            color = _BAND_COLORS[i % len(_BAND_COLORS)]
            f.configure(style='TFrame')

            mat_var   = tk.StringVar(value=ldata['material'])
            thick_var = tk.StringVar(value=str(ldata['thickness']))
            self._layer_vars.append((mat_var, thick_var))

            ttk.Label(f, text=str(i + 1), width=3,
                      anchor='center').pack(side=tk.LEFT)

            cb = ttk.Combobox(f, textvariable=mat_var,
                               values=MATERIAL_KEYS, width=15, state='readonly')
            cb.pack(side=tk.LEFT, padx=3)

            ttk.Entry(f, textvariable=thick_var, width=9).pack(side=tk.LEFT, padx=3)

            ttk.Button(f, text='\u2715', width=3, style='Del.TButton',
                        command=lambda i=i: self._delete_layer(i)).pack(side=tk.LEFT)

        self._scroll_canvas.configure(
            scrollregion=self._scroll_canvas.bbox('all'))

    def _sync_layer_data(self):
        """
        Flush the current UI StringVar values back into self._layer_data.
        Must be called before any operation that calls _rebuild_layer_ui(),
        otherwise live edits the user has typed are discarded when the
        widget tree is torn down and recreated from stale _layer_data values.
        """
        for i, (mat_var, thick_var) in enumerate(self._layer_vars):
            self._layer_data[i]['material']  = mat_var.get()
            raw = thick_var.get().strip()
            try:
                self._layer_data[i]['thickness'] = float(raw)
            except ValueError:
                pass  # keep the old value; _read_layers will catch it on Run

    def _add_layer_clicked(self):
        self._sync_layer_data()
        self._layer_data.append({'material': MATERIAL_KEYS[0], 'thickness': 0.5})
        self._rebuild_layer_ui()

    def _delete_layer(self, idx: int):
        if len(self._layer_data) <= 1:
            messagebox.showwarning("Layer Stack",
                                   "Cannot delete: at least one layer is required.")
            return
        self._sync_layer_data()
        del self._layer_data[idx]
        self._rebuild_layer_ui()

    def _read_layers(self) -> List[PlateLayer]:
        """Parse layer stack from UI variables; raises ValueError on bad input."""
        layers = []
        for i, (mat_var, thick_var) in enumerate(self._layer_vars):
            mat = mat_var.get()
            raw = thick_var.get().strip()
            try:
                thick = float(raw)
                if thick <= 0.0:
                    raise ValueError("must be > 0")
            except ValueError as exc:
                raise ValueError(
                    f"Layer {i+1} ('{mat}'): invalid thickness '{raw}' -- {exc}")
            layers.append(PlateLayer(mat, thick))
        return layers

    # ----------------------------------------------------------
    #  Analysis Execution  (threaded)
    # ----------------------------------------------------------

    def _on_run(self):
        # Parse global parameters
        try:
            q_flux  = float(self._q_var.get())
            t_total = float(self._t_var.get())
            T_init  = float(self._T0_var.get())
        except ValueError as exc:
            messagebox.showerror("Input Error", f"Invalid parameter value:\n{exc}")
            return

        if q_flux <= 0.0:
            messagebox.showerror("Input Error",
                                 "Heat flux must be positive.")
            return
        if t_total <= 0.0:
            messagebox.showerror("Input Error",
                                 "Duration must be positive.")
            return

        try:
            layers = self._read_layers()
        except ValueError as exc:
            messagebox.showerror("Layer Error", str(exc))
            return

        # Launch solver
        self._run_btn.config(state=tk.DISABLED)
        self._prog.start(10)
        self._log_clear()
        self._log(f"q\" = {q_flux} BTU/(ft^2*s)  |  t = {t_total} s  "
                  f"|  T0 = {T_init} degF\n"
                  f"Layers: {' / '.join(f'{l.material_key} {l.thickness} in' for l in layers)}\n")

        threading.Thread(
            target=self._solver_thread,
            args=(layers, q_flux, t_total, T_init),
            daemon=True
        ).start()

        self.root.after(150, self._poll)

    def _solver_thread(self, layers, q_flux, t_total, T_init):
        """Runs in background thread; puts result on queue."""
        old_stdout = sys.stdout
        buf = io.StringIO()
        sys.stdout = buf
        try:
            solver = MultilayerHeatConductionSolver(layers, THERMAL_PROPERTIES)
            z_in, T_final, T_history = solver.solve_transient(
                q_flux=q_flux,
                t_total=t_total,
                T_initial=T_init,
                report_interval=0.5
            )
            sys.stdout = old_stdout
            log_txt = buf.getvalue()

            # Append layer temperature table
            cap = io.StringIO()
            sys.stdout = cap
            solver.print_layer_temperatures(z_in, T_final, T_init)
            sys.stdout = old_stdout
            log_txt += cap.getvalue()

            self._result_q.put(
                ('ok', solver, z_in, T_final, T_history,
                 T_init, q_flux, t_total, log_txt)
            )
        except Exception:
            sys.stdout = old_stdout
            self._result_q.put(('err', tb.format_exc()))

    def _poll(self):
        """Check result queue; reschedule if not ready."""
        try:
            msg = self._result_q.get_nowait()
        except queue.Empty:
            self.root.after(150, self._poll)
            return

        self._prog.stop()
        self._run_btn.config(state=tk.NORMAL)

        if msg[0] == 'err':
            self._log('\n[ERROR]\n' + msg[1])
            messagebox.showerror("Solver Error",
                                 msg[1][:500] + ('...' if len(msg[1]) > 500 else ''))
            return

        _, solver, z_in, T_final, T_history, T_init, q_flux, t_total, log_txt = msg
        self._log(log_txt)
        self._update_all_plots(solver, z_in, T_final, T_history,
                                T_init, q_flux, t_total)

    # ----------------------------------------------------------
    #  Plotting
    # ----------------------------------------------------------

    def _update_all_plots(self, solver, z_in, T_final, T_history,
                           T_init, q_flux, t_total):
        self._plot_evolution(solver, z_in, T_history, T_init, q_flux, t_total)
        self._plot_delta_T  (solver, z_in, T_final,   T_init, t_total)
        self._plot_contour  (solver, z_in, T_history,          t_total)
        self._notebook.select(0)

    def _shade_layers(self, ax, solver: MultilayerHeatConductionSolver):
        """
        Overlay semi-transparent material bands and dashed boundary lines.
        Must be called AFTER the data is plotted so y-limits are known.
        """
        ylim  = ax.get_ylim()
        z_lo  = 0.0
        for i, (z_hi, lyr) in enumerate(
                zip(solver.layer_boundaries, solver.layers)):
            color = _BAND_COLORS[i % len(_BAND_COLORS)]
            ax.axvspan(z_lo, z_hi, color=color, alpha=0.12, zorder=0)
            ax.axvline(z_hi, color='#777', linestyle='--',
                       linewidth=0.8, alpha=0.6, zorder=1)
            mid   = (z_lo + z_hi) / 2.0
            label = lyr.material_key.replace('_', '\n')
            ax.text(mid, ylim[0] + 0.96 * (ylim[1] - ylim[0]),
                    label, ha='center', va='top',
                    fontsize=7, color='#444', fontstyle='italic', zorder=2)
            z_lo  = z_hi
        ax.set_ylim(ylim)  # restore; band text can push limits

    def _plot_evolution(self, solver, z_in, T_history, T_init,
                         q_flux, t_total):
        ax = self._ax1
        ax.clear()
        ax.set_facecolor('#fafafa')

        times  = sorted(T_history.keys())
        n_show = min(8, len(times))
        idxs   = np.round(np.linspace(0, len(times) - 1, n_show)).astype(int)
        cmap   = plt.cm.plasma

        for j, idx in enumerate(idxs):
            t     = times[idx]
            color = cmap(j / max(n_show - 1, 1))
            ax.plot(z_in, T_history[t],
                    color=color, linewidth=2.0, zorder=3,
                    label=f't = {t:.2f} s')

        ax.axhline(T_init, color='k', linestyle=':', linewidth=1.0,
                   alpha=0.5, label=f'T\u2080 = {T_init:.0f}\u00b0F')

        ax.set_xlabel('Depth from Heated Surface (in)', fontsize=10)
        ax.set_ylabel('Temperature (\u00b0F)', fontsize=10)
        ax.set_title(
            f'Temperature Evolution\n'
            f'q" = {q_flux:.0f} BTU/(ft\u00b2\u00b7s),  '
            f't_total = {t_total:.2f} s',
            fontsize=10, fontweight='bold')
        ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlim(z_in[0], z_in[-1])

        self._shade_layers(ax, solver)
        self._fig1.tight_layout()
        self._cv1.draw()

    def _plot_delta_T(self, solver, z_in, T_final, T_init, t_total):
        ax = self._ax2
        ax.clear()
        ax.set_facecolor('#fafafa')

        dT = T_final - T_init
        ax.plot(z_in, dT, color='#c62828', linewidth=2.5, zorder=3)
        ax.fill_between(z_in, 0.0, dT,
                        alpha=0.25, color='#ef5350', zorder=2)
        ax.axhline(0.0, color='k', linewidth=0.8)

        ax.set_xlabel('Depth from Heated Surface (in)', fontsize=10)
        ax.set_ylabel('Temperature Rise \u0394T (\u00b0F)', fontsize=10)
        ax.set_title(
            f'Final Temperature Rise\n'
            f't = {t_total:.2f} s  |  '
            f'\u0394T_surface = {dT[0]:.1f}\u00b0F  |  '
            f'\u0394T_back = {dT[-1]:.1f}\u00b0F',
            fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlim(z_in[0], z_in[-1])

        self._shade_layers(ax, solver)
        self._fig2.tight_layout()
        self._cv2.draw()

    def _plot_contour(self, solver, z_in, T_history, t_total):
        ax = self._ax3
        ax.clear()

        # Safely remove old colorbar before redrawing
        if self._cbar3 is not None:
            try:
                self._cbar3.remove()
            except Exception:
                pass
            self._cbar3 = None

        times    = sorted(T_history.keys())
        T_matrix = np.array([T_history[t] for t in times])  # shape (N_t, N_z)

        # np.meshgrid(z_in, times) -> shapes (N_t, N_z) each
        Z_mesh, Time_mesh = np.meshgrid(z_in, times)
        cf = ax.contourf(Time_mesh, Z_mesh, T_matrix, levels=25, cmap='hot')
        self._cbar3 = self._fig3.colorbar(cf, ax=ax, label='Temperature (\u00b0F)')

        # Layer boundary lines (horizontal on depth axis)
        for bnd in solver.layer_boundaries:
            ax.axhline(bnd, color='cyan', linestyle='--',
                       linewidth=0.9, alpha=0.75)

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Depth from Surface (in)', fontsize=10)
        ax.set_title('Temperature Contour  (Depth vs. Time)',
                     fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(times[0], times[-1])
        self._fig3.tight_layout()
        self._cv3.draw()

    # ----------------------------------------------------------
    #  Log Utilities
    # ----------------------------------------------------------

    def _log_clear(self):
        self._log_widget.config(state='normal')
        self._log_widget.delete('1.0', tk.END)
        self._log_widget.config(state='disabled')

    def _log(self, msg: str):
        self._log_widget.config(state='normal')
        self._log_widget.insert(tk.END, msg)
        self._log_widget.see(tk.END)
        self._log_widget.config(state='disabled')
        self.root.update_idletasks()


# ============================================================
#  ENTRY POINT
# ============================================================

def main():
    root = tk.Tk()
    try:
        root.tk.call('tk', 'scaling', 1.2)
    except Exception:
        pass
    app = ThermalSolverGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
