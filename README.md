# Multilayer Transient Heat Conduction Solver

A Python application for 1-D transient heat conduction analysis through multilayer walls, with an interactive GUI for configuring material stacks, load parameters, and visualization of results.

---

## Physics

The solver integrates the 1-D transient heat equation using an explicit Forward-Time Central-Space (FTCS) finite-difference scheme:

$$\rho c_p \frac{\partial T}{\partial t} = \frac{\partial}{\partial z}\left[k(z)\,\frac{\partial T}{\partial z}\right]$$

where $k(z)$, $\rho(z)$, and $c_p(z)$ are spatially varying to accommodate multiple material layers.

### Boundary Conditions

| Surface | Type | Condition |
|---|---|---|
| Front (heated) | Neumann | Prescribed heat flux $q''$ |
| Back | Neumann | Insulated ($q = 0$, $\partial T/\partial z = 0$) |

### Interface Conductivity

At material interfaces, the inter-nodal conductivity is computed using the harmonic mean to enforce heat flux continuity (Patankar, *Numerical Heat Transfer and Fluid Flow*, 1980):

$$k_{i+\frac{1}{2}} = \frac{2\,k_i\,k_{i+1}}{k_i + k_{i+1}}$$

### Stability

The explicit scheme is conditionally stable. The time step is automatically constrained by the Von Neumann criterion applied to the most diffusive layer in the stack:

$$\text{Fo} = \frac{\alpha_{\min}\,\Delta t}{\Delta z^2} \leq 0.5 \qquad \Rightarrow \qquad \Delta t_{\max} = \frac{\Delta z^2}{2\,\alpha_{\min}}$$

where $\alpha = k / (\rho c_p)$ is the thermal diffusivity. The solver reports the actual Fourier number to the log window on every run and flags any instability.

### Units

All inputs and outputs use **U.S. Customary units**:

| Quantity | Unit |
|---|---|
| Thermal conductivity $k$ | BTU/(hr·ft·°F) |
| Density $\rho$ | lb/ft³ |
| Specific heat $c_p$ | BTU/(lb·°F) |
| Thermal diffusivity $\alpha$ | ft²/s |
| Heat flux $q''$ | BTU/(ft²·s) |
| Temperature | °F |
| Layer thickness | in |
| Time | s |

---

## Features

- **Configurable layer stack** -- add, remove, and reorder material layers; each layer has an independently selectable material and thickness
- **Parameter editing** -- heat flux, duration, and initial temperature are all editable before each run
- **Stability check** -- Fourier number computed and reported automatically; unstable configurations are flagged in the log
- **Three output plots**, each in its own tab with a full Matplotlib navigation toolbar (zoom, pan, save to PNG):
  - Temperature evolution at multiple time snapshots
  - Final temperature rise profile $\Delta T(z,\, t_{\text{total}})$ with filled area
  - Full space-time temperature contour map $T(z,\, t)$
- **Layer boundary table** -- tabular printout of temperature and $\Delta T$ at every material interface, logged after each run

---

## Material Database

Three materials are included by default. Additional materials can be added by extending the `THERMAL_PROPERTIES` dictionary at the top of the source file.

| Key | Material | $k$ [BTU/(hr·ft·°F)] | $\rho$ [lb/ft³] | $c_p$ [BTU/(lb·°F)] |
|---|---|---|---|---|
| `A572_GR50` | Structural Steel (A572 Gr. 50) | 34.9 | 490.0 | 0.120 |
| `C95900` | Aluminum Bronze (C95900) | 41.1 | 440.6 | 0.090 |
| `ASTM_B21_H02` | Naval Brass (ASTM B21 H02) | 35.0 | 535.0 | 0.090 |

To add a new material, append an entry to `THERMAL_PROPERTIES` before the `MATERIAL_KEYS` line:

```python
THERMAL_PROPERTIES['316L_SS'] = ThermalProperties(
    k=9.4,      # BTU/(hr·ft·°F)
    rho=499.0,  # lb/ft³
    cp=0.12     # BTU/(lb·°F)
)
```

---

## Installation

### Requirements

- Python 3.10 or later
- NumPy
- Matplotlib
- Tkinter (included in the Python standard library; Linux users may need to install `python3-tk` separately)

---

## Usage

1. **Set global parameters** in the "Load & Duration" panel:
   - Heat flux $q''$ [BTU/(ft²·s)]
   - Total duration $t$ [s]
   - Initial temperature $T_0$ [°F]

2. **Configure the layer stack** from heated surface to back surface:
   - Select material from the dropdown for each layer
   - Enter layer thickness in inches
   - Use **+ Add Layer** to append a new layer
   - Use the **✕** button on any row to delete that layer

3. **Click Run Analysis.** The solver log will display the grid parameters, Fourier stability number, time-stepping progress, and the final layer boundary temperature table.

4. **Review results** across the three plot tabs. Each plot supports zoom, pan, and export via the Matplotlib toolbar.

### Default Configuration

The application launches pre-loaded with the following four-layer stack as a reference case:

| Layer | Material | Thickness (in) |
|---|---|---|
| 1 (heated surface) | A572_GR50 | 0.750 |
| 2 | C95900 | 0.093 |
| 3 | ASTM_B21_H02 | 0.530 |
| 4 (back surface) | C95900 | 0.093 |

Default load: $q'' = 375$ BTU/(ft²·s), $t = 7.2$ s, $T_0 = 70$°F.

---

## Code Structure

```
Temp3_GUI.py
│
├── PlateLayer              dataclass -- material key + thickness (in)
├── ThermalProperties       dataclass -- k, rho, cp; computes alpha
├── THERMAL_PROPERTIES      dict -- material database
│
├── MultilayerHeatConductionSolver
│   ├── __init__            accepts layer list and material dict
│   ├── _setup_grid         builds nodal property arrays at 50 nodes/in
│   ├── solve_transient     FTCS time integration; returns z, T, T_history
│   └── print_layer_temperatures   tabular boundary report to stdout
│
└── ThermalSolverGUI        Tkinter application class
    ├── _build_controls     left panel -- parameters, layer stack, log
    ├── _build_plot_area    right panel -- three tabbed Matplotlib canvases
    ├── _sync_layer_data    flushes live StringVar values back to _layer_data
    ├── _rebuild_layer_ui   tears down and recreates layer row widgets
    ├── _on_run             validates inputs, launches solver thread
    ├── _solver_thread      runs solve_transient in background thread
    ├── _poll               checks result queue; updates plots on completion
    ├── _plot_evolution     Tab 1 -- temperature snapshots vs. depth
    ├── _plot_delta_T       Tab 2 -- final delta-T profile
    └── _plot_contour       Tab 3 -- space-time contour map
```

---

## Limitations and Assumptions

- **Temperature-independent material properties.** $k$, $\rho$, and $c_p$ are treated as constants. For problems spanning large temperature ranges, tabulated properties should be reviewed.
- **Constant heat flux.** The front-surface boundary condition does not support time-varying or spatially varying flux. A time-varying $q''(t)$ can be implemented by modifying the front-surface BC inside the time loop in `solve_transient`.
- **Insulated back surface.** The rear boundary is assumed adiabatic. A convective or radiation BC can be substituted at the last node if needed.
- **Explicit scheme stability.** For very thin layers or highly diffusive materials, the stable time step may become extremely small, leading to long runtimes. Reducing `nodes_per_inch` in `_setup_grid` will coarsen the grid and increase $\Delta t_{\max}$ at the cost of spatial resolution.

---

## References

- Patankar, S.V., *Numerical Heat Transfer and Fluid Flow*, Hemisphere Publishing, 1980.
- Incropera, F.P. et al., *Fundamentals of Heat and Mass Transfer*, 7th ed., Wiley, 2011.
- AISC 360-22, *Specification for Structural Steel Buildings*, American Institute of Steel Construction, 2022.
- ASTM A572/A572M, *Standard Specification for High-Strength Low-Alloy Columbium-Vanadium Structural Steel*.
- ASTM B21/B21M, *Standard Specification for Naval Brass Rod, Bar, and Shapes*.
