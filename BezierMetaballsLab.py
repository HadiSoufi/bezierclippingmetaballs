#!/usr/bin/env python3
"""
Bezier-clipped metaballs lab.

An interactive harness for BezierClippedMetaballs.py: drop circles, drag a ray
through them, and watch the density curve and every Bezier-clipping iteration
update live. Then render a full frame and compare against ground truth.

Run:
    py -3.10 BezierMetaballsLab.py
    py -3.10 BezierMetaballsLab.py --selftest

Scene controls
    left click empty space ....... drop a circle
    drag circle body ............. move it
    drag circle rim .............. resize it
    right click circle ........... delete it
    drag the round handle ........ move the ray origin
    drag the arrow handle ........ aim the ray
    mouse wheel .................. zoom
    middle drag .................. pan
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import bezier_metaballs_core as C  # noqa: E402


# ------------------------------------------------------------------ palette

# Two full palettes. The app follows the system theme rather than hardcoding a
# light one -- otherwise a system-wide dark-mode forcer (Windhawk, etc.) darkens
# the widget backgrounds while our explicit text colours stay dark, and the
# result is unreadable. The mid-tone marks (ball/ray/hit) are chosen to read on
# both backgrounds; only the backgrounds, grid, faint lines and text differ.
PALETTES = {
    "light": dict(
        BG="#f7f7fa", CANVAS_BG="#ffffff", PANEL="#ffffff", BORDER="#d0d0dc",
        GRID="#e8e8ef", AXIS="#c8c8d4",
        TXT="#141414", TXT_DIM="#6b6b6b", TXT_GOOD="#127338", TXT_BAD="#c0392b",
        FAN="#c3d3ec", FOV="#93aad4", META="#8a68bd",
        STRIP_MARK_IN="#111111", STRIP_MARK_OUT="#ffffff", STRIP_BG="#f6f6fa",
        PLOT_BG="#ffffff", PLOT_FG="#222222", PLOT_GRID="#dddddd",
        BALL="#7d5ba6", BALL_SEL="#d1495b", RAY="#2a6fdb", TICK="#d1495b",
        HIT_POC="#e07a00", HIT_REF="#1a9850", ISO="#1a9850", HANDLE="#2a6fdb",
    ),
    "dark": dict(
        BG="#232327", CANVAS_BG="#17171b", PANEL="#1e1e22", BORDER="#3a3a42",
        GRID="#2c2c33", AXIS="#454550",
        TXT="#e6e6ea", TXT_DIM="#9a9aa2", TXT_GOOD="#4ec27e", TXT_BAD="#ef6a5c",
        FAN="#33435f", FOV="#3f5a86", META="#a888e0",
        STRIP_MARK_IN="#ffffff", STRIP_MARK_OUT="#000000", STRIP_BG="#202024",
        PLOT_BG="#1a1a1e", PLOT_FG="#d0d0d6", PLOT_GRID="#33333a",
        BALL="#b28fe0", BALL_SEL="#f06a7c", RAY="#5b95ec", TICK="#f06a7c",
        HIT_POC="#ffa640", HIT_REF="#4ec27e", ISO="#4ec27e", HANDLE="#5b95ec",
    ),
}


def _system_prefers_dark():
    """Windows 'Apps use dark theme' from the registry. Best-effort."""
    try:
        import winreg
        k = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        val, _ = winreg.QueryValueEx(k, "AppsUseLightTheme")
        winreg.CloseKey(k)
        return val == 0
    except Exception:
        return False


def apply_theme(name):
    """Publish one palette's entries as module globals used across the UI."""
    globals().update(PALETTES[name])
    globals()["THEME"] = name


# Start on whatever the system reports; the toggle overrides it live.
apply_theme("dark" if _system_prefers_dark() else "light")

HANDLE_PX = 7.0
RIM_PX = 8.0


# ------------------------------------------------------------------ helpers

def fmt(v, n=4):
    if v is None:
        return "-"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    return f"{v:.{n}f}"


class Scene:
    """Circles plus a 2D camera. `origin` is the camera position, `angle` is
    where it looks, `fov` is the angular width of the fan of rays it casts."""

    def __init__(self):
        self.spheres = [list(s) for s in C.PRESETS["Blob: two merging"]["spheres"]]
        self.origin = np.array([-1.5, 0.0])
        self.angle = 0.0
        self.fov = math.radians(70.0)

    @property
    def direction(self):
        return np.array([math.cos(self.angle), math.sin(self.angle)])

    def ray_angles(self, n):
        """Angles of the fan, left edge to right edge."""
        if n <= 1:
            return np.array([self.angle])
        return self.angle + np.linspace(-self.fov / 2, self.fov / 2, n)

    def to_dict(self):
        return {"spheres": [list(map(float, s)) for s in self.spheres],
                "origin": [float(self.origin[0]), float(self.origin[1])],
                "angle": float(self.angle),
                "fov": float(self.fov)}

    def load(self, d):
        self.spheres = [list(map(float, s)) for s in d["spheres"]]
        self.origin = np.array([float(d["origin"][0]), float(d["origin"][1])])
        self.angle = float(d.get("angle", 0.0))
        self.fov = float(d.get("fov", math.radians(70.0)))


# =========================================================================
# Main window
# =========================================================================

class Lab(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bezier-Clipped Metaballs Lab")
        self.geometry("1560x940")
        self.configure(bg=BG)
        self.minsize(1180, 720)

        self.scene = Scene()

        # view transform
        self.view_cx = 3.0
        self.view_cy = 0.0
        self.view_scale = 78.0

        # tunables
        self.threshold = tk.DoubleVar(value=0.2)
        self.max_iters = tk.IntVar(value=20)
        self.show_truth = tk.BooleanVar(value=True)
        self.show_meta = tk.BooleanVar(value=True)
        self.iter_index = tk.IntVar(value=0)
        self.render_mode = tk.StringVar(value="Shaded")

        # camera
        self.cam_fov = tk.DoubleVar(value=70.0)
        self.cam_rays = tk.IntVar(value=129)
        self.show_fan = tk.BooleanVar(value=True)
        self.sel_ray = tk.IntVar(value=64)

        # fan results, recomputed whenever the scene changes
        self.fan_t = None          # (n,) hit distance, nan for a miss
        self.fan_p = None          # (n,2) hit points
        self.fan_n = None          # (n,2) surface normals
        self.fan_ref = None        # (n,) reference distance, for the diff mode
        self.fan_angles = None

        # state
        self.trace = None
        self.ref_t = None
        self.ref_p = None
        self._drag = None
        self._iso_cache_key = None
        self._iso_cache = []
        self._plot_pending = False

        self._build_ui()
        self.after(60, self.retrace)

    # ---------------------------------------------------------------- theme

    def _apply_widget_theme(self):
        """(Re)style every ttk widget class from the active palette. Explicit
        fg AND bg on everything, so a system dark-mode forcer cannot leave one
        of them at a clashing default."""
        s = self.style
        self.configure(bg=BG)
        s.configure(".", background=BG, foreground=TXT, fieldbackground=PANEL,
                    bordercolor=BORDER, lightcolor=BG, darkcolor=BG)
        for cls in ("TFrame", "TLabel", "TCheckbutton", "TLabelframe",
                    "TLabelframe.Label", "TRadiobutton"):
            s.configure(cls, background=BG, foreground=TXT)
        s.map("TCheckbutton", foreground=[("!disabled", TXT)])
        s.configure("TButton", background=PANEL, foreground=TXT)
        s.map("TButton", background=[("active", BORDER)])
        s.configure("TSpinbox", fieldbackground=PANEL, foreground=TXT,
                    background=PANEL, arrowcolor=TXT, bordercolor=BORDER)
        s.map("TSpinbox", fieldbackground=[("readonly", PANEL)],
              foreground=[("readonly", TXT)])
        # Comboboxes here are state=readonly, which is styled through the
        # "readonly" state map, NOT the base config -- that is why they stayed
        # light. The map covers the closed field, its text, and the arrow box.
        s.configure("TCombobox", fieldbackground=PANEL, foreground=TXT,
                    background=PANEL, arrowcolor=TXT, bordercolor=BORDER,
                    selectbackground=PANEL, selectforeground=TXT)
        s.map("TCombobox",
              fieldbackground=[("readonly", PANEL), ("disabled", BG)],
              foreground=[("readonly", TXT), ("disabled", TXT_DIM)],
              background=[("readonly", PANEL), ("active", BORDER)],
              selectbackground=[("readonly", PANEL)],
              selectforeground=[("readonly", TXT)],
              arrowcolor=[("readonly", TXT)])
        s.configure("Hdr.TLabel", font=("Segoe UI", 9, "bold"),
                    background=BG, foreground=TXT)
        s.configure("Mono.TLabel", font=("Consolas", 9),
                    background=BG, foreground=TXT_DIM)
        # Popup list of every combobox, and defaults for classic tk widgets.
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", TXT)
        self.option_add("*TCombobox*Listbox.selectBackground", HANDLE)
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    def _set_theme(self, name):
        """Switch the whole UI to a palette and repaint."""
        apply_theme(name)
        self._apply_widget_theme()
        # Widgets that carry their own colours rather than a ttk style.
        self.canvas.configure(bg=CANVAS_BG, highlightbackground=BORDER)
        self.strip.configure(bg=STRIP_BG, highlightbackground=BORDER)
        self.readout.configure(bg=PANEL, fg=TXT, insertbackground=TXT)
        self.fig.set_facecolor(BG)
        self._iter_n = -1                 # force the iteration bar to rebuild
        self.retrace()

    def _poll_system_theme(self):
        """Follow the OS light/dark setting without a manual control.

        There is no dependency-free change event for this, so poll the registry
        once a second; it is a single cheap read and only repaints on a change.
        """
        want = "dark" if _system_prefers_dark() else "light"
        if want != THEME:
            self._set_theme(want)
        self.after(1000, self._poll_system_theme)

    # ---------------------------------------------------------------- build

    def _build_ui(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self._apply_widget_theme()

        root = ttk.Frame(self, padding=6)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=3, minsize=560)
        root.columnconfigure(1, weight=4, minsize=520)
        root.columnconfigure(2, weight=0, minsize=290)
        root.rowconfigure(0, weight=1)

        # ---- scene ----
        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Scene  —  click to drop a circle, drag rim to resize, "
                             "right-click to delete", style="Hdr.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 3))

        self.canvas = tk.Canvas(left, bg=CANVAS_BG, highlightthickness=1,
                                highlightbackground=BORDER, width=620, height=560)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self.redraw_scene())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right)
        self.canvas.bind("<ButtonPress-2>", self.on_mid_press)
        self.canvas.bind("<B2-Motion>", self.on_mid_drag)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

        ttk.Label(left, text="Camera image  —  one pixel per ray; click to inspect "
                             "that ray", style="Hdr.TLabel").grid(
            row=2, column=0, sticky="w", pady=(6, 2))
        self.strip = tk.Canvas(left, height=54, bg=STRIP_BG, highlightthickness=1,
                               highlightbackground=BORDER)
        self.strip.grid(row=3, column=0, sticky="ew")
        self.strip.bind("<Button-1>", self.on_strip_click)
        self.strip.bind("<B1-Motion>", self.on_strip_click)
        self.strip.bind("<Configure>", lambda e: self.draw_strip())
        self._strip_img = None

        self.status = ttk.Label(left, text="", style="Mono.TLabel")
        self.status.grid(row=4, column=0, sticky="w", pady=(4, 0))

        # ---- plots ----
        mid = ttk.Frame(root)
        mid.grid(row=0, column=1, sticky="nsew", padx=(0, 6))
        mid.rowconfigure(1, weight=1)
        mid.columnconfigure(0, weight=1)

        ttk.Label(mid, text="Density along the ray", style="Hdr.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 3))

        self.fig = Figure(figsize=(6.0, 6.4), dpi=100, facecolor=BG)
        self.ax_full = self.fig.add_subplot(211)
        self.ax_iter = self.fig.add_subplot(212)
        self.fig.subplots_adjust(left=0.11, right=0.98, top=0.93,
                                 bottom=0.08, hspace=0.34)
        self.plot = FigureCanvasTkAgg(self.fig, master=mid)
        self.plot.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # One button per clip iteration, filling the width. A slider was the
        # wrong control: a root takes 4-6 iterations, so its whole travel was a
        # few pixels; a segmented bar makes each step a real target.
        step = ttk.Frame(mid)
        step.grid(row=2, column=0, sticky="ew", pady=(8, 2))
        step.columnconfigure(0, weight=1)
        ttk.Label(step, text="Clip iteration  —  step through the root search",
                  style="Hdr.TLabel").grid(row=0, column=0, sticky="w")
        self.iter_bar = ttk.Frame(step)
        self.iter_bar.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._iter_buttons = []
        self.bind_all("<Left>", lambda e: self._step_iter(-1))
        self.bind_all("<Right>", lambda e: self._step_iter(+1))

        # ---- controls ----
        right = ttk.Frame(root)
        right.grid(row=0, column=2, sticky="nsew")
        right.rowconfigure(6, weight=1)
        self._build_controls(right)

    def _build_controls(self, parent):
        r = 0

        pre = ttk.LabelFrame(parent, text="Scene", padding=6)
        pre.grid(row=r, column=0, sticky="ew", pady=(0, 6)); r += 1
        pre.columnconfigure(0, weight=1)
        self.preset_var = tk.StringVar(value="Blob: two merging")
        cb = ttk.Combobox(pre, textvariable=self.preset_var, state="readonly",
                          values=list(C.PRESETS.keys()))
        cb.grid(row=0, column=0, columnspan=2, sticky="ew")
        cb.bind("<<ComboboxSelected>>", self.on_preset)
        btns = ttk.Frame(pre); btns.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        for i, (txt, cmd) in enumerate([("Fit", self.fit_view), ("Clear", self.clear_scene),
                                        ("Save", self.save_scene), ("Load", self.load_scene)]):
            ttk.Button(btns, text=txt, width=7, command=cmd).grid(row=0, column=i, padx=1)

        par = ttk.LabelFrame(parent, text="Parameters", padding=6)
        par.grid(row=r, column=0, sticky="ew", pady=(0, 6)); r += 1
        par.columnconfigure(1, weight=1)
        self._spin(par, 0, "Threshold", self.threshold, 0.01, 3.0, 0.01)
        self._spin(par, 1, "Max iters", self.max_iters, 1, 60, 1)
        ttk.Label(par, text="Tolerance").grid(row=2, column=0, sticky="w", pady=1)
        self.tol_var = tk.StringVar(value="1e-6")
        tolcb = ttk.Combobox(par, textvariable=self.tol_var, state="readonly",
                             width=8, values=["1e-2", "1e-3", "1e-4", "1e-6", "1e-9"])
        tolcb.grid(row=2, column=1, sticky="ew", padx=(4, 0), pady=1)
        tolcb.bind("<<ComboboxSelected>>", lambda e: self.retrace())

        cam = ttk.LabelFrame(parent, text="Camera", padding=6)
        cam.grid(row=r, column=0, sticky="ew", pady=(0, 6)); r += 1
        cam.columnconfigure(1, weight=1)
        self._spin(cam, 0, "FOV (deg)", self.cam_fov, 5.0, 175.0, 5.0,
                   cb=self.on_fov)
        self._spin(cam, 1, "Rays", self.cam_rays, 9, 1025, 16)
        ttk.Label(cam, text="Shading").grid(row=2, column=0, sticky="w", pady=(3, 0))
        ttk.Combobox(cam, textvariable=self.render_mode, state="readonly", width=12,
                     values=["Shaded", "Depth", "Normal", "Difference"]
                     ).grid(row=2, column=1, sticky="ew", padx=(4, 0), pady=(3, 0))

        ov = ttk.LabelFrame(parent, text="Overlays", padding=6)
        ov.grid(row=r, column=0, sticky="ew", pady=(0, 6)); r += 1
        ttk.Checkbutton(ov, text="Draw isosurface",
                        variable=self.show_fan, command=self.redraw_scene).grid(
            row=0, column=0, sticky="w")
        ttk.Checkbutton(ov, text="Lone-ball surface (dashed)",
                        variable=self.show_meta, command=self.redraw_scene).grid(
            row=1, column=0, sticky="w")
        ttk.Checkbutton(ov, text="Ground truth in density plot",
                        variable=self.show_truth, command=self.retrace).grid(
            row=2, column=0, sticky="w")

        insp = ttk.LabelFrame(parent, text="Ray inspector  (the selected pixel)",
                              padding=6)
        insp.grid(row=r, column=0, sticky="ew"); r += 1
        insp.columnconfigure(0, weight=1)
        self.readout = tk.Text(insp, width=40, height=17, font=("Consolas", 8),
                               wrap="none", bg=PANEL, fg=TXT, insertbackground=TXT,
                               relief="flat", spacing1=0, spacing3=0)
        self.readout.grid(row=0, column=0, sticky="ew")

        # Let leftover vertical space go here rather than stretching the panel.
        spacer = ttk.Frame(parent)
        spacer.grid(row=r, column=0, sticky="nsew"); r += 1
        parent.rowconfigure(r - 1, weight=1)

    def _spin(self, parent, row, label, var, lo, hi, step, fmtstr=None, cb=None):
        cb = cb or self.retrace
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        kw = {"format": fmtstr} if fmtstr else {}
        s = ttk.Spinbox(parent, from_=lo, to=hi, increment=step, textvariable=var,
                        width=10, command=cb, **kw)
        s.grid(row=row, column=1, sticky="ew", padx=(4, 0), pady=1)
        s.bind("<Return>", lambda e: cb())
        s.bind("<FocusOut>", lambda e: cb())

    def on_fov(self):
        self.scene.fov = math.radians(float(self.cam_fov.get()))
        self.retrace()

    # ------------------------------------------------------------ transform

    def w2s(self, p):
        w = self.canvas.winfo_width() or 1
        h = self.canvas.winfo_height() or 1
        return (w / 2 + (p[0] - self.view_cx) * self.view_scale,
                h / 2 - (p[1] - self.view_cy) * self.view_scale)

    def s2w(self, x, y):
        w = self.canvas.winfo_width() or 1
        h = self.canvas.winfo_height() or 1
        return np.array([self.view_cx + (x - w / 2) / self.view_scale,
                         self.view_cy - (y - h / 2) / self.view_scale])

    def fit_view(self):
        pts = []
        for cx, cy, R in self.scene.spheres:
            pts += [(cx - R, cy - R), (cx + R, cy + R)]
        pts.append(tuple(self.scene.origin))
        if not pts:
            return
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2; cy = (min(ys) + max(ys)) / 2
        spanx = max(max(xs) - min(xs), 1e-3); spany = max(max(ys) - min(ys), 1e-3)
        w = self.canvas.winfo_width() or 600
        h = self.canvas.winfo_height() or 560
        self.view_cx, self.view_cy = cx, cy
        self.view_scale = 0.82 * min(w / spanx, h / spany)
        self.redraw_scene()

    # ---------------------------------------------------------------- input

    def _pick(self, x, y):
        """Return ('origin',None) / ('tip',None) / ('body',i) / ('rim',i) / None."""
        p = self.s2w(x, y)
        ox, oy = self.w2s(self.scene.origin)
        if math.hypot(x - ox, y - oy) <= HANDLE_PX + 3:
            return ("origin", None)
        tip = self.scene.origin + self.scene.direction * (110.0 / self.view_scale)
        tx, ty = self.w2s(tip)
        if math.hypot(x - tx, y - ty) <= HANDLE_PX + 3:
            return ("tip", None)
        for i in range(len(self.scene.spheres) - 1, -1, -1):
            cx, cy, R = self.scene.spheres[i]
            d = math.hypot(p[0] - cx, p[1] - cy) * self.view_scale
            rpx = R * self.view_scale
            if abs(d - rpx) <= RIM_PX:
                return ("rim", i)
            if d < rpx:
                return ("body", i)
        return None

    def on_press(self, e):
        hit = self._pick(e.x, e.y)
        if hit is None:
            p = self.s2w(e.x, e.y)
            self.scene.spheres.append([float(p[0]), float(p[1]), 1.0])
            self._drag = ("rim", len(self.scene.spheres) - 1, None)
            self.clear_render()
            self.retrace()
            return
        kind, idx = hit
        if kind == "body":
            p = self.s2w(e.x, e.y)
            cx, cy, _ = self.scene.spheres[idx]
            self._drag = ("body", idx, (cx - p[0], cy - p[1]))
        else:
            self._drag = (kind, idx, None)
        self.redraw_scene()

    def on_drag(self, e):
        if self._drag is None:
            return
        kind, idx, aux = self._drag
        p = self.s2w(e.x, e.y)
        if kind == "body":
            self.scene.spheres[idx][0] = float(p[0] + aux[0])
            self.scene.spheres[idx][1] = float(p[1] + aux[1])
        elif kind == "rim":
            cx, cy, _ = self.scene.spheres[idx]
            self.scene.spheres[idx][2] = max(0.05, float(math.hypot(p[0] - cx, p[1] - cy)))
        elif kind == "origin":
            self.scene.origin = p
        elif kind == "tip":
            d = p - self.scene.origin
            if np.linalg.norm(d) > 1e-9:
                self.scene.angle = float(math.atan2(d[1], d[0]))
        self.retrace()

    def on_release(self, e):
        if self._drag is not None:
            self._drag = None
            # Now that the drag is over: full-resolution contour, and the
            # matplotlib panels catch up.
            self.clear_render()
            self.retrace()

    def on_right(self, e):
        hit = self._pick(e.x, e.y)
        if hit and hit[0] in ("body", "rim"):
            del self.scene.spheres[hit[1]]
            self.clear_render()
            self.retrace()

    def on_mid_press(self, e):
        self._pan = (e.x, e.y, self.view_cx, self.view_cy)

    def on_mid_drag(self, e):
        if not hasattr(self, "_pan"):
            return
        x0, y0, cx0, cy0 = self._pan
        self.view_cx = cx0 - (e.x - x0) / self.view_scale
        self.view_cy = cy0 + (e.y - y0) / self.view_scale
        self.redraw_scene()

    def on_wheel(self, e):
        before = self.s2w(e.x, e.y)
        self.view_scale *= 1.1 if e.delta > 0 else 1 / 1.1
        self.view_scale = min(max(self.view_scale, 4.0), 4000.0)
        after = self.s2w(e.x, e.y)
        self.view_cx += before[0] - after[0]
        self.view_cy += before[1] - after[1]
        self.clear_render()
        self.redraw_scene()

    def on_preset(self, _e=None):
        p = C.PRESETS[self.preset_var.get()]
        self.scene.spheres = [list(s) for s in p["spheres"]]
        self.scene.origin = np.array([float(p["origin"][0]), float(p["origin"][1])])
        self.scene.angle = float(p.get("angle", 0.0))
        self.clear_render()
        self.fit_view()
        self.retrace()

    def clear_scene(self):
        self.scene.spheres = []
        self.clear_render()
        self.retrace()

    def save_scene(self):
        fn = filedialog.asksaveasfilename(defaultextension=".json",
                                          filetypes=[("Scene JSON", "*.json")])
        if fn:
            with open(fn, "w") as f:
                json.dump(self.scene.to_dict(), f, indent=2)

    def load_scene(self):
        fn = filedialog.askopenfilename(filetypes=[("Scene JSON", "*.json")])
        if not fn:
            return
        try:
            with open(fn) as f:
                self.scene.load(json.load(f))
        except Exception as ex:
            messagebox.showerror("Load failed", str(ex))
            return
        self.clear_render()
        self.fit_view()
        self.retrace()

    def set_iter(self, k):
        self.iter_index.set(int(k))
        self.refresh_iter_buttons()
        self.draw_plots()
        self.fill_readout()

    def _step_iter(self, delta):
        """Move between clip iterations (arrows / prev-next buttons)."""
        # Don't steal arrow keys from a text field or dropdown that has focus.
        w = self.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox, tk.Text)):
            return
        n = len(self.trace.iters) if self.trace else 0
        if n:
            self.set_iter(min(max(self.iter_index.get() + delta, 0), n - 1))

    def refresh_iter_buttons(self):
        """Rebuild the per-iteration segmented bar to match the current trace.

        Layout: [<] [1][2][3][4] [>], the numbered cells stretching to fill the
        width so each is an easy target and the whole bar reads as one control.
        """
        n = len(self.trace.iters) if self.trace else 0
        cur = min(max(self.iter_index.get(), 0), max(n - 1, 0))
        bar = self.iter_bar

        # Rebuild the widgets only when the number of iterations changes; a
        # drag retraces every frame and rebuilding buttons each time is wasteful.
        if n != getattr(self, "_iter_n", -1):
            for b in self._iter_buttons:
                b.destroy()
            self._iter_buttons = []
            for col in range(bar.grid_size()[0]):
                bar.columnconfigure(col, weight=0, uniform="")
            self._iter_n = n

            if n == 0:
                lbl = tk.Label(bar, text="no root search on this ray  "
                                         "(segment already resolved)",
                               font=("Segoe UI", 9), fg=TXT_DIM, bg=BG, anchor="w")
                lbl.grid(row=0, column=0, sticky="w")
                self._iter_buttons.append(lbl)
            else:
                prev = tk.Button(bar, text="‹", font=("Segoe UI", 11), width=2,
                                 relief="flat", bd=0,
                                 command=lambda: self._step_iter(-1))
                prev.grid(row=0, column=0, sticky="ns", padx=(0, 3))
                self._iter_buttons.append(prev)
                for k in range(n):
                    b = tk.Button(bar, text=str(k + 1), font=("Segoe UI", 10),
                                  relief="flat", bd=0, pady=6,
                                  command=lambda kk=k: self.set_iter(kk))
                    b.grid(row=0, column=k + 1, sticky="nsew", padx=1)
                    bar.columnconfigure(k + 1, weight=1, uniform="iter")
                    self._iter_buttons.append(b)
                nxt = tk.Button(bar, text="›", font=("Segoe UI", 11), width=2,
                                relief="flat", bd=0,
                                command=lambda: self._step_iter(+1))
                nxt.grid(row=0, column=n + 1, sticky="ns", padx=(3, 0))
                self._iter_buttons.append(nxt)

        # Recolour every time (selection, and after a theme switch).
        if n == 0:
            if self._iter_buttons:
                self._iter_buttons[0].configure(fg=TXT_DIM, bg=BG)
            return
        for arrow in (self._iter_buttons[0], self._iter_buttons[-1]):
            arrow.configure(bg=PANEL, fg=TXT,
                            activebackground=BORDER, activeforeground=TXT)
        for k in range(n):
            b = self._iter_buttons[k + 1]     # skip the leading '‹'
            on = (k == cur)
            b.configure(bg=HANDLE if on else PANEL,
                        fg="#ffffff" if on else TXT,
                        activebackground=HANDLE if on else BORDER,
                        activeforeground="#ffffff" if on else TXT,
                        font=("Segoe UI", 10, "bold" if on else "normal"))

    # ---------------------------------------------------------------- trace

    def tracer(self):
        return C.trace_ray_fixed

    def retrace(self, *_):
        sph = self.scene.spheres
        thr = float(self.threshold.get())

        self.compute_fan()
        self.sync_selected_ray()
        pd = self.probe_direction()

        if not sph:
            self.trace = None
            self.ref_t = self.ref_p = None
        else:
            try:
                self.trace = self.tracer()(
                    sph, self.scene.origin, pd,
                    thr, int(self.max_iters.get()),
                    float(self.tol_var.get()))
            except Exception as ex:
                self.trace = None
                self.status.configure(text=f"trace raised: {type(ex).__name__}: {ex}")
            self.ref_t, self.ref_p, _ = C.reference_hit(sph, self.scene.origin, pd, thr)

        n = len(self.trace.iters) if self.trace else 0
        if self.iter_index.get() > max(n - 1, 0):
            self.iter_index.set(max(n - 1, 0))
        self.refresh_iter_buttons()

        self.redraw_scene()
        self.draw_strip()
        self.fill_readout()
        self.schedule_plots()

    # ------------------------------------------------------------ camera fan

    def compute_fan(self, n=None):
        """Trace the camera's fan of rays.

        The set of hit points across the fan IS the visible silhouette -- the
        part of the isosurface this camera can see. Tracing it with the same
        code path as a single ray means the silhouette is a picture of the
        algorithm's output, not a separate rendering of the field.
        """
        sph = self.scene.spheres
        if n is None:
            # A fan costs ~220 us/ray, so keep it cheap while dragging and
            # fill in detail once the mouse settles.
            n = 49 if self._drag is not None else int(self.cam_rays.get())
        n = max(int(n), 3)

        if not sph:
            self.fan_t = self.fan_p = self.fan_n = self.fan_ref = None
            self.fan_angles = None
            return

        angles = self.scene.ray_angles(n)
        trace = self.tracer()
        thr = float(self.threshold.get())
        iters = int(self.max_iters.get())
        tol = float(self.tol_var.get())

        t = np.full(n, np.nan)
        pts = np.full((n, 2), np.nan)
        nrm = np.zeros((n, 2))

        for i, a in enumerate(angles):
            d = np.array([math.cos(a), math.sin(a)])
            try:
                tr = trace(sph, self.scene.origin, d, thr, iters, tol,
                           record=False)
            except Exception:
                continue
            if tr.surface is None or tr.surface < 0:
                continue
            t[i] = tr.surface
            pts[i] = self.scene.origin + tr.surface * d
            nrm[i] = C.field_normal(pts[i], sph)

        self.fan_angles = angles
        self.fan_t = t
        self.fan_p = pts
        self.fan_n = nrm

        if self.render_mode.get() == "Difference":
            ref = np.full(n, np.nan)
            for i, a in enumerate(angles):
                d = [math.cos(a), math.sin(a)]
                rt, _, _ = C.reference_hit(sph, self.scene.origin, d, thr)
                if rt is not None:
                    ref[i] = rt
            self.fan_ref = ref
        else:
            self.fan_ref = None

    def draw_strip(self):
        """The 1D image this 2D camera produces -- one pixel per ray."""
        cv = self.strip
        cv.delete("all")
        w = cv.winfo_width() or 1
        h = cv.winfo_height() or 1
        if self.fan_t is None or len(self.fan_t) == 0:
            self._strip_img = None
            return

        t = self.fan_t
        n = len(t)
        hit = np.isfinite(t)
        bg_rgb = [int(STRIP_BG[i:i + 2], 16) for i in (1, 3, 5)]
        rgb = np.full((1, n, 3), 0, dtype=np.uint8)
        rgb[:] = bg_rgb                                  # background = strip bg

        mode = self.render_mode.get()
        if mode == "Difference" and self.fan_ref is not None:
            ref = self.fan_ref
            both = hit & np.isfinite(ref)
            rgb[0, hit & ~np.isfinite(ref)] = [214, 40, 40]     # invented
            rgb[0, ~hit & np.isfinite(ref)] = [30, 60, 200]     # lost
            if both.any():
                e = np.abs(t[both] - ref[both])
                hi = max(float(np.percentile(e, 98)), 1e-9)
                rgb[0, both] = _colormap(np.clip(e / hi, 0, 1))
        elif mode == "Depth" and hit.any():
            d = t[hit]
            lo, hi = float(d.min()), float(d.max())
            if hi <= lo:
                hi = lo + 1e-6
            rgb[0, hit] = _colormap(1.0 - (d - lo) / (hi - lo))
        elif mode == "Normal" and hit.any():
            nn = self.fan_n[hit]
            rgb[0, hit] = np.stack([
                (0.5 + 0.5 * nn[:, 0]) * 255,
                (0.5 + 0.5 * nn[:, 1]) * 255,
                np.full(nn.shape[0], 200.0)], axis=-1).astype(np.uint8)
        elif hit.any():                                  # Shaded
            light = np.array([-0.4, 0.85])
            light = light / np.linalg.norm(light)
            lam = np.clip(self.fan_n[hit] @ light, 0, 1)
            shade = 0.18 + 0.82 * lam
            base = np.array([125.0, 91.0, 166.0])
            rgb[0, hit] = np.clip(shade[:, None] * base[None, :]
                                  + 55.0 * (lam ** 12)[:, None], 0, 255).astype(np.uint8)

        try:
            img = tk.PhotoImage(data=_to_ppm(rgb), format="PPM")
            zx = max(int(w // n), 1)
            img = img.zoom(zx, 1).zoom(1, max(h, 1))
            self._strip_img = img
            cv.create_image(0, 0, image=img, anchor="nw")
        except Exception:
            self._strip_img = None

        # mark the inspected ray
        k = min(max(int(self.sel_ray.get()), 0), n - 1)
        x = (k + 0.5) / n * w
        cv.create_line(x, 0, x, h, fill=STRIP_MARK_OUT, width=3)
        cv.create_line(x, 0, x, h, fill=STRIP_MARK_IN, width=1)

    def on_strip_click(self, e):
        if self.fan_t is None:
            return
        n = len(self.fan_t)
        w = self.strip.winfo_width() or 1
        self.sel_ray.set(min(max(int(e.x / w * n), 0), n - 1))
        self.sync_selected_ray()
        self.retrace()

    def sync_selected_ray(self):
        """Point the inspected ray at the currently selected fan column."""
        if self.fan_angles is None or len(self.fan_angles) == 0:
            return
        k = min(max(int(self.sel_ray.get()), 0), len(self.fan_angles) - 1)
        self._probe_angle = float(self.fan_angles[k])

    def probe_direction(self):
        a = getattr(self, "_probe_angle", None)
        if a is None:
            return self.scene.direction
        return np.array([math.cos(a), math.sin(a)])

    def schedule_plots(self):
        """Coalesce matplotlib redraws.

        Flushing the two subplots costs ~165 ms, which is by far the most
        expensive thing a drag can trigger -- an order of magnitude more than
        the trace itself. So during a drag the plots are left alone and the
        cheap Tk scene canvas keeps up at full rate; they refresh on release.
        Outside a drag, redraws coalesce onto the idle handler so a burst of
        parameter changes still only costs one flush.
        """
        if self._drag is not None:
            return
        if not self._plot_pending:
            self._plot_pending = True
            self.after_idle(self._flush_plots)

    def _flush_plots(self):
        self._plot_pending = False
        self.draw_plots()

    # ----------------------------------------------------------- scene draw

    def redraw_scene(self):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width() or 1
        h = c.winfo_height() or 1

        self._draw_grid(w, h)

        for i, (cx, cy, R) in enumerate(self.scene.spheres):
            x0, y0 = self.w2s((cx - R, cy + R))
            x1, y1 = self.w2s((cx + R, cy - R))
            sel = self._drag is not None and self._drag[1] == i
            # Outline only. A filled circle paints over everything drawn
            # underneath it -- which is why the isosurface overlay looked
            # broken: it was being drawn, then covered by the next ball.
            c.create_oval(x0, y0, x1, y1, outline=BALL_SEL if sel else BALL,
                          width=2)

            # Where this ball's surface would be if it were alone. The gap
            # between this and the traced silhouette IS the blending.
            if self.show_meta.get():
                rm = C.meta_radius(R, float(self.threshold.get()))
                if rm > 0:
                    m0 = self.w2s((cx - rm, cy + rm))
                    m1 = self.w2s((cx + rm, cy - rm))
                    c.create_oval(*m0, *m1, outline=META, width=1, dash=(3, 3))
            px, py = self.w2s((cx, cy))
            c.create_line(px - 4, py, px + 4, py, fill=BALL)
            c.create_line(px, py - 4, px, py + 4, fill=BALL)
            c.create_text(px, py - 11, text=str(i), fill=BALL,
                          font=("Segoe UI", 8, "bold"))

        self._draw_ray(w, h)

    def _draw_grid(self, w, h):
        c = self.canvas
        step = 1.0
        while step * self.view_scale < 44:
            step *= 2
        while step * self.view_scale > 150:
            step /= 2
        p0 = self.s2w(0, h); p1 = self.s2w(w, 0)
        x = math.floor(p0[0] / step) * step
        while x <= p1[0]:
            sx, _ = self.w2s((x, 0))
            c.create_line(sx, 0, sx, h, fill=GRID)
            x += step
        y = math.floor(p0[1] / step) * step
        while y <= p1[1]:
            _, sy = self.w2s((0, y))
            c.create_line(0, sy, w, sy, fill=GRID)
            y += step
        ax, ay = self.w2s((0, 0))
        c.create_line(0, ay, w, ay, fill=AXIS)
        c.create_line(ax, 0, ax, h, fill=AXIS)

    def _draw_ray(self, w, h):
        c = self.canvas
        o = self.scene.origin
        far = max(w, h) / self.view_scale * 2

        # ---- field of view ----
        for sgn in (-1, 1):
            a = self.scene.angle + sgn * self.scene.fov / 2
            e = o + np.array([math.cos(a), math.sin(a)]) * far
            c.create_line(*self.w2s(o), *self.w2s(e), fill=FOV, width=1)

        # ---- the fan, and the silhouette its hits trace out ----
        if self.fan_t is not None and self.show_fan.get():
            t = self.fan_t
            n = len(t)
            step = max(n // 48, 1)
            for i in range(0, n, step):
                a = self.fan_angles[i]
                dv = np.array([math.cos(a), math.sin(a)])
                end = self.fan_p[i] if np.isfinite(t[i]) else o + dv * far
                c.create_line(*self.w2s(o), *self.w2s(end),
                              fill=FAN, width=1)

            # Join adjacent hits into the visible surface outline. A big jump
            # in depth between neighbours is a real silhouette edge, not a
            # surface, so break the polyline there.
            runs = []
            cur = []
            for i in range(n):
                if np.isfinite(t[i]) and (not cur or abs(t[i] - t[i - 1]) < 0.35):
                    cur.append(self.fan_p[i])
                else:
                    if len(cur) > 1:
                        runs.append(cur)
                    cur = [self.fan_p[i]] if np.isfinite(t[i]) else []
            if len(cur) > 1:
                runs.append(cur)
            for run in runs:
                pts = []
                for p in run:
                    pts.extend(self.w2s(p))
                c.create_line(*pts, fill=HIT_POC, width=3, capstyle="round")

        d = self.probe_direction()
        c.create_line(*self.w2s(o), *self.w2s(o + d * far), fill=RAY, width=2)

        tr = self.trace
        if tr:
            for br in tr.balls:
                if br.missed:
                    continue
                for t in (br.start, br.end):
                    px, py = self.w2s(o + d * t)
                    c.create_line(px, py - 6, px, py + 6, fill=TICK, width=2)

        if self.ref_p is not None:
            x, y = self.w2s(self.ref_p)
            c.create_oval(x - 6, y - 6, x + 6, y + 6, outline=HIT_REF, width=2)
            c.create_text(x, y - 14, text="ref", fill=HIT_REF,
                          font=("Segoe UI", 8, "bold"))

        if tr and tr.hit_point is not None:
            x, y = self.w2s(tr.hit_point)
            c.create_line(x - 7, y - 7, x + 7, y + 7, fill=HIT_POC, width=3)
            c.create_line(x - 7, y + 7, x + 7, y - 7, fill=HIT_POC, width=3)
            if tr.hit_point is not None and self.fan_n is not None:
                nrm = C.field_normal(tr.hit_point, self.scene.spheres)
                tipn = tr.hit_point + nrm * (34.0 / self.view_scale)
                c.create_line(*self.w2s(tr.hit_point), *self.w2s(tipn),
                              fill=HIT_POC, width=2, arrow="last")

        # ---- camera body: drag to move, drag the nose to aim ----
        ox, oy = self.w2s(o)
        fd = self.scene.direction
        perp = np.array([-fd[1], fd[0]])
        s = 13.0 / self.view_scale
        poly = []
        for pt in (o - fd * s * 0.7 + perp * s * 0.75,
                   o + fd * s * 1.15,
                   o - fd * s * 0.7 - perp * s * 0.75):
            poly.extend(self.w2s(pt))
        c.create_polygon(*poly, fill=HANDLE, outline="white", width=2)
        c.create_text(ox, oy + 22, text="camera", fill=HANDLE,
                      font=("Segoe UI", 8, "bold"))

        tip = o + fd * (110.0 / self.view_scale)
        tx, ty = self.w2s(tip)
        c.create_line(ox, oy, tx, ty, fill=HANDLE, width=1, dash=(3, 3))
        c.create_oval(tx - HANDLE_PX, ty - HANDLE_PX, tx + HANDLE_PX, ty + HANDLE_PX,
                      fill="white", outline=HANDLE, width=2)

    def _iso_segments(self, coarse=False):
        """Contour of the true isosurface, cached on scene + threshold + grid.

        A drag busts this cache on every motion event, so it uses the in-house
        marching squares (~4 ms) rather than matplotlib's contour(), which costs
        a fixed ~15 ms per call whatever the grid size.
        """
        n = 110 if coarse else 190
        key = (tuple(map(tuple, self.scene.spheres)),
               round(float(self.threshold.get()), 6), n)
        if key == self._iso_cache_key:
            return self._iso_cache
        segs = []
        try:
            xs = [c for c, _, r in self.scene.spheres]
            ys = [c for _, c, r in self.scene.spheres]
            rs = [r for _, _, r in self.scene.spheres]
            pad = max(rs) * 1.15
            gx = np.linspace(min(xs) - pad, max(xs) + pad, n)
            gy = np.linspace(min(ys) - pad, max(ys) + pad, n)
            X, Y = np.meshgrid(gx, gy)
            Z = C.field(np.stack([X, Y], axis=-1), self.scene.spheres)
            segs = C.iso_contour(Z, gx, gy, float(self.threshold.get()))
        except Exception:
            segs = []
        self._iso_cache_key = key
        self._iso_cache = segs
        return segs

    # ------------------------------------------------------------ plotting

    def draw_plots(self):
        tr = self.trace
        thr = float(self.threshold.get())

        self.fig.set_facecolor(BG)

        ax = self.ax_full
        ax.clear()
        self._style_ax(ax)
        ax.set_title("Composite density over the selected range", fontsize=9,
                     color=PLOT_FG)
        ax.set_xlabel("distance along ray", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.grid(alpha=0.25, color=PLOT_GRID)
        ax.axhline(thr, color=ISO, lw=1.2, label=f"threshold {thr:g}")

        if self.show_truth.get() and self.scene.spheres:
            far = C._scene_far(self.scene.spheres, self.scene.origin)
            t, f = C.sample_along_ray(self.scene.spheres, self.scene.origin,
                                      self.scene.direction, 0.0, far, 500)
            ax.plot(t, f, color=PLOT_FG, lw=1.6, label="ground truth  Σf")

        if tr and tr.selected_curve is not None and tr.curve_t1 != tr.curve_t0:
            ctrl = tr.selected_curve
            ts = np.linspace(0, 1, 200)
            pts = C.eval_bezier(ctrl, ts)
            world = tr.curve_t0 + pts[:, 0] * (tr.curve_t1 - tr.curve_t0)
            ax.plot(world, pts[:, 1], color=BALL, lw=2.0,
                    label="Bezier composite (selected range)")
            cw = tr.curve_t0 + ctrl[:, 0] * (tr.curve_t1 - tr.curve_t0)
            ax.plot(cw, ctrl[:, 1], "o--", color=BALL, ms=3.5, lw=0.8, alpha=0.55)

        if self.ref_t is not None:
            ax.axvline(self.ref_t, color=HIT_REF, lw=1.2, ls=":", label="reference root")
        if tr and tr.surface is not None:
            ax.axvline(tr.surface, color=HIT_POC, lw=1.2, ls="--", label="traced surface")
        self._legend(ax)

        ax = self.ax_iter
        ax.clear()
        self._style_ax(ax)
        ax.grid(alpha=0.25, color=PLOT_GRID)
        ax.axhline(thr, color=ISO, lw=1.2)
        ax.set_xlabel("normalized curve parameter", fontsize=8)
        ax.set_ylabel("density", fontsize=8)

        if tr and tr.iters:
            k = min(self.iter_index.get(), len(tr.iters) - 1)
            it = tr.iters[k]
            ax.set_title(f"Clip iteration {k + 1} of {len(tr.iters)}   "
                         f"t_min={it.t_min:.4f}  t_max={it.t_max:.4f}",
                         fontsize=9, color=PLOT_FG)
            ctrl = it.curve
            ts = np.linspace(0, 1, 200)
            pts = C.eval_bezier(ctrl, ts)
            hull = C.convex_hull(ctrl)
            if len(hull) >= 3:
                ax.fill(hull[:, 0], hull[:, 1], color=ISO, alpha=0.22,
                        label="control-point hull")
            ax.plot(ctrl[:, 0], ctrl[:, 1], "o--", color=BALL_SEL, ms=4, lw=0.9,
                    label="control polygon")
            ax.plot(pts[:, 0], pts[:, 1], color=PLOT_FG, lw=2, label="curve")
            for tv, col in ((it.t_min, RAY), (it.t_max, HIT_POC)):
                ax.axvline(tv, color=col, lw=1.4)
                ax.plot([tv], [thr], "o", color=col, ms=6)
            self._legend(ax)
            ax.set_xlim(-0.05, 1.05)
        else:
            ax.set_title("Clip iterations — no surface found on this ray",
                         fontsize=9, color=PLOT_FG)
            ax.set_xlim(0, 1)

        self.plot.draw_idle()

    def _style_ax(self, ax):
        """Theme a matplotlib axis: background, spines, ticks, labels."""
        ax.set_facecolor(PLOT_BG)
        for sp in ax.spines.values():
            sp.set_color(PLOT_GRID)
        ax.tick_params(labelsize=7, colors=PLOT_FG)
        ax.xaxis.label.set_color(PLOT_FG)
        ax.yaxis.label.set_color(PLOT_FG)

    def _legend(self, ax):
        leg = ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
        if leg:
            leg.get_frame().set_facecolor(PLOT_BG)
            leg.get_frame().set_edgecolor(PLOT_GRID)
            for txt in leg.get_texts():
                txt.set_color(PLOT_FG)

    # ------------------------------------------------------------- readout

    def fill_readout(self):
        """Everything the tracer did on the selected ray, in the order it did it."""
        t = self.readout
        t.configure(state="normal")
        t.delete("1.0", "end")
        tr = self.trace
        o = self.scene.origin

        n_rays = 0 if self.fan_t is None else len(self.fan_t)
        k = min(max(self.sel_ray.get(), 0), max(n_rays - 1, 0))
        ang = (math.degrees(self.fan_angles[k])
               if self.fan_angles is not None and n_rays else
               math.degrees(self.scene.angle))

        t.insert("end", f"pixel {k}/{n_rays}   bearing {ang:+.1f} deg   "
                        f"from ({o[0]:+.2f},{o[1]:+.2f})\n", "dim")

        if tr is None:
            t.insert("end", "\nno circles in the scene\n")
            t.tag_configure("dim", foreground="#888888")
            t.configure(state="disabled")
            self.status.configure(text="")
            return

        # 1 -- where the ray crosses each circle, and the curve that implies
        t.insert("end", "\nCHORDS   entry->exit distance, a=(L/2R)^2\n", "h")
        rows = 0
        for b in tr.balls:
            if b.missed:
                continue
            rows += 1
            t.insert("end", f" {b.index}: {b.start:7.3f}->{b.end:<7.3f} "
                            f"R{b.radius:.2f}  a {b.a:.4f}  peak {max(b.control_y):.3f}\n")
        if rows == 0:
            t.insert("end", " ray misses every circle\n", "dim")

        # 2 -- the sweep
        t.insert("end", "\nSEGMENTS   split at every chord end\n", "h")
        if not tr.ranges_split:
            t.insert("end", " none\n", "dim")
        for i, rg in enumerate(tr.ranges_split):
            balls = ",".join(str(b) for b in rg.ball_indices) or "-"
            hit_here = (i == tr.selected_index)
            t.insert("end", f" {rg.start:7.3f}->{rg.end:<7.3f} balls {balls}"
                            f"{'   <- crossing' if hit_here else ''}\n",
                     ("good",) if hit_here else ())

        # 3 -- the root search, one line per clip iteration
        t.insert("end", "\nROOT SEARCH   bracket on the segment, 0..1\n", "h")
        if not tr.iters:
            t.insert("end", " already above threshold at segment start\n"
                     if tr.found_surface else
                     " no segment reached the threshold\n", "dim")
        else:
            for i, it in enumerate(tr.iters):
                sel = ">" if i == self.iter_index.get() else " "
                t.insert("end", f"{sel}{i + 1}: [{it.t_min:.5f}, {it.t_max:.5f}]"
                                f"  width {it.t_max - it.t_min:.1e}\n")

        # 4 -- the answer, and how far it is from the truth
        t.insert("end", "\nRESULT   world units\n", "h")
        if tr.surface is not None:
            nrm = C.field_normal(tr.hit_point, self.scene.spheres)
            t.insert("end", f" surface at {tr.surface:.6f} along the ray\n")
            t.insert("end", f" point   ({tr.hit_point[0]:+.4f}, {tr.hit_point[1]:+.4f})"
                            f"   normal ({nrm[0]:+.2f},{nrm[1]:+.2f})\n")
        else:
            t.insert("end", " no surface on this ray\n")

        if tr.surface is not None and self.ref_t is not None:
            err = tr.surface - self.ref_t
            ok = abs(err) < 1e-6
            t.insert("end", f" brute-force solver says {self.ref_t:.6f}\n", "dim")
            t.insert("end", f" off by {err:+.1e}  -> "
                            f"{'matches' if ok else 'DISAGREES'}\n",
                     "good" if ok else "bad")
        elif tr.surface is None and self.ref_t is not None:
            t.insert("end", f" brute-force solver says {self.ref_t:.6f}\n", "dim")
            t.insert("end", " MISSED a real surface\n", "bad")
        elif tr.surface is not None and self.ref_t is None:
            t.insert("end", " brute-force solver finds nothing here\n", "dim")
            t.insert("end", " INVENTED a surface\n", "bad")

        t.tag_configure("h", font=("Consolas", 8, "bold"), foreground=TXT)
        t.tag_configure("dim", foreground=TXT_DIM)
        t.tag_configure("bad", foreground=TXT_BAD)
        t.tag_configure("good", foreground=TXT_GOOD)

        # Size the box to what it is actually showing. A fixed height is either
        # padded with dead space or silently clipping the result line.
        lines = int(t.index("end-1c").split(".")[0])
        t.configure(height=min(max(lines, 8), 30), state="disabled")

        # The line under the camera image describes the IMAGE. Per-ray numbers
        # live in the inspector above -- repeating them here said nothing new.
        if self.fan_t is not None and n_rays:
            hits = int(np.isfinite(self.fan_t).sum())
            self.status.configure(
                text=f"{n_rays} rays   {hits} hit the surface   "
                     f"{n_rays - hits} background   |   "
                     f"{len(self.scene.spheres)} circles")
        else:
            self.status.configure(text="")

    # -------------------------------------------------------------- render

    def clear_render(self):
        """Drop cached camera results so the next retrace recomputes them."""
        self.fan_t = self.fan_p = self.fan_n = self.fan_ref = None
        self.fan_angles = None


# ------------------------------------------------------------- image utils

def _colormap(v):
    """Simple perceptual-ish ramp, v in [0,1] -> uint8 RGB."""
    v = np.clip(np.nan_to_num(v), 0, 1)
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140],
                      [94, 201, 98], [253, 231, 37]], dtype=float)
    x = v * (len(stops) - 1)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def _to_ppm(rgb):
    h, w = rgb.shape[:2]
    return b"P6\n%d %d\n255\n" % (w, h) + np.ascontiguousarray(
        rgb.astype(np.uint8)).tobytes()


# =========================================================================
# Self-test
# =========================================================================

def selftest():
    """Characterize the PoC against ground truth.

    Each check declares the result it is EXPECTED to produce today, so this
    doubles as a regression harness: PASS/KNOWN means nothing moved, FAIL means
    something that used to work broke, and FIXED means a known defect is gone
    and the expectation should be updated.
    """
    rng = np.random.default_rng(20260809)
    rows = []

    def row(name, ok, detail, expect=True, ref=""):
        rows.append((name, ok, detail, expect, ref))

    # 1. Center-ray peak: a = 1, so the curve must peak at f(0) = 1.
    for mode, label, expect, ref in (
            (C.A_POC, "a = (det/R)^2   [PoC]", False, "O1"),
            (C.A_PAPER, "a = (L/2R)^2    [paper]", True, "")):
        R = 1.3
        m = C.Metaball([3.0, 0.0], R, 3.0 - R, 3.0 + R, (2 * R) ** 2)
        cps = m.density_control_points(mode)
        peak = float(C.eval_bezier(cps, [0.5])[0, 1])
        ok = abs(peak - 1.0) < 1e-9
        row(f"center-ray peak == 1.0   {label}", ok,
            f"a={m.a_value(mode):.6f}  peak={peak:.9f}", expect, ref)

    # 2. Bezier control points vs the paper's eq. (10).
    worst = 0.0
    for _ in range(200):
        a = float(rng.uniform(0.02, 1.0))
        R = float(rng.uniform(0.4, 3.0))
        h = R * math.sqrt(max(1.0 - a, 0.0))
        L = 2.0 * R * math.sqrt(a)
        m = C.Metaball([0.0, h], R, -L / 2, L / 2, L * L)
        got = m.density_control_points(C.A_PAPER)[:, 1]
        worst = max(worst, float(np.max(np.abs(got - C.paper_control_points(a)))))
    row("control points match paper eq.(10)", worst < 1e-12, f"max abs diff {worst:.3e}")

    # 3. Bezier curve vs the field sampled directly along the ray.
    #    This is the check that the whole method rests on: the degree-6 Bezier
    #    really is the field restricted to the chord.
    worst = 0.0
    for _ in range(200):
        R = float(rng.uniform(0.5, 2.5))
        h = float(rng.uniform(0.0, 0.97)) * R
        cx = float(rng.uniform(1.0, 5.0))
        sph = [[cx, h, R]]
        half = math.sqrt(max(R * R - h * h, 0.0))
        start, end = cx - half, cx + half
        m = C.Metaball([cx, h], R, start, end, (2 * half) ** 2)
        cps = m.density_control_points(C.A_PAPER)
        ts = np.linspace(0, 1, 60)
        curve = C.eval_bezier(cps, ts)
        world = start + curve[:, 0] * (end - start)
        pts = np.stack([world, np.zeros_like(world)], axis=-1)
        truth = C.field(pts, sph)
        worst = max(worst, float(np.max(np.abs(curve[:, 1] - truth))))
    row("Bezier curve == field along ray", worst < 1e-9, f"max abs diff {worst:.3e}")

    # 4/5. Root accuracy, split by whether any two chords along the ray overlap.
    #      This is the load-bearing distinction: with no overlap the method is
    #      pure single-ball Bezier clipping, and it should be exact to within
    #      the clip tolerance. Everything else exercises the range splitter and
    #      the compositing step.
    def chords_overlap(sph):
        ch = []
        for cx, cy, R in sph:
            if abs(cy) >= R:
                continue
            half = math.sqrt(R * R - cy * cy)
            ch.append((cx - half, cx + half))
        ch.sort()
        return any(ch[i][1] > ch[i + 1][0] for i in range(len(ch) - 1))

    def sample_scenes(want_overlap, count):
        out = []
        while len(out) < count:
            k = int(rng.integers(1, 4))
            sph = [[float(rng.uniform(1.5, 5.0)), float(rng.uniform(-1.2, 1.2)),
                    float(rng.uniform(0.5, 1.6))] for _ in range(k)]
            sph.sort(key=lambda s: s[0])
            if chords_overlap(sph) == want_overlap:
                out.append(sph)
        return out

    def score(scenes, v):
        errs, miss, false_hit = [], 0, 0
        for sph in scenes:
            ref, _, _ = C.reference_hit(sph, [0, 0], [1, 0], 0.2)
            tr = C.trace_ray(sph, [0, 0], [1, 0], 0.2, variants=v)
            if ref is None and tr.surface is not None:
                false_hit += 1
            elif ref is not None and tr.surface is None:
                miss += 1
            elif ref is not None and tr.surface is not None:
                errs.append(abs(tr.surface - ref))
        e = np.array(errs) if errs else np.array([np.nan])
        return e, miss, false_hit

    disjoint = sample_scenes(False, 200)
    overlapping = sample_scenes(True, 200)
    v_fixed = C.Variants(C.A_PAPER, C.HULL_POC, C.SURFACE_SCALED)

    for label, v, scenes, bound, expect, ref in (
            ("no chord overlap  [PoC as written]",
             C.Variants(), disjoint, 1e-3, False, "O1+O5"),
            ("no chord overlap  [a=paper, surface=mapped]",
             v_fixed, disjoint, 1e-3, True, ""),
            ("chords overlap    [a=paper, surface=mapped]",
             v_fixed, overlapping, 1e-3, False, "O2/O3/O4/O9")):
        e, miss, false_hit = score(scenes, v)
        ok = bool(np.isfinite(e).any()) and np.nanmax(e) < bound \
            and miss == 0 and false_hit == 0
        row(f"root vs bisection: {label}", ok,
            f"n={int(np.isfinite(e).sum())} max={np.nanmax(e):.3e} "
            f"mean={np.nanmean(e):.3e} miss={miss} false={false_hit}", expect, ref)

    # 6. The compositing step in isolation: rays that graze two balls, where
    #    neither ball alone reaches the threshold so the surface exists only
    #    because the two densities sum.
    errs, miss, false_hit, n_comp = [], 0, 0, 0
    for h in np.arange(0.58, 0.96, 0.02):
        for sep in np.arange(0.3, 2.0, 0.1):
            R = 1.3
            sph = [[3.0 - sep / 2, h * R, R], [3.0 + sep / 2, h * R, R]]
            tr = C.trace_ray(sph, [0, 0], [1, 0], 0.2, variants=v_fixed)
            if tr.selected_range is None or tr.selected_range.ball_count < 2:
                continue
            n_comp += 1
            ref, _, _ = C.reference_hit(sph, [0, 0], [1, 0], 0.2)
            if ref is None and tr.surface is not None:
                false_hit += 1
            elif ref is not None and tr.surface is None:
                miss += 1
            elif ref is not None and tr.surface is not None:
                errs.append(abs(tr.surface - ref))
    e = np.array(errs) if errs else np.array([np.nan])
    ok = bool(errs) and np.nanmax(e) < 1e-3 and miss == 0 and false_hit == 0
    row("composited range decides the hit", ok,
        f"n={n_comp} max={np.nanmax(e):.3e} mean={np.nanmean(e):.3e} "
        f"miss={miss} false={false_hit}", False, "O2/O3/O4")

    # 6. Order independence.
    bad = 0
    for _ in range(60):
        sph = [[float(rng.uniform(1.5, 5.0)), float(rng.uniform(-1.0, 1.0)),
                float(rng.uniform(0.6, 1.5))] for _ in range(3)]
        fwd = C.trace_ray(sph, [0, 0], [1, 0], 0.2)
        rev = C.trace_ray(list(reversed(sph)), [0, 0], [1, 0], 0.2)
        a_, b_ = fwd.surface, rev.surface
        if (a_ is None) != (b_ is None) or (a_ is not None and abs(a_ - b_) > 1e-9):
            bad += 1
    row("result independent of input order", bad == 0,
        f"{bad}/60 scenes differ", False, "O8")

    # 6b. The corrected implementation, over the cases that broke the PoC.
    def fixed_sweep(nballs, nrays, inside=False, angled=False):
        errs, miss, false_hit, its = [], 0, 0, []
        for _ in range(nrays):
            k = nballs if isinstance(nballs, int) else int(rng.integers(*nballs))
            sph = [[float(rng.uniform(1.0, 5.0)), float(rng.uniform(-1.5, 1.5)),
                    float(rng.uniform(0.4, 1.7))] for _ in range(k)]
            if inside:
                o = np.array([float(rng.uniform(1.0, 5.0)),
                              float(rng.uniform(-1.5, 1.5))])
            else:
                o = np.array([-1.0, float(rng.uniform(-2.5, 2.5))])
            th = float(rng.uniform(-0.6, 0.6)) if angled else 0.0
            d = [math.cos(th), math.sin(th)]
            ref, _, _ = C.reference_hit(sph, o, d, 0.2)
            tr = C.trace_ray_fixed(sph, o, d, 0.2)
            if ref is None and tr.surface is not None:
                false_hit += 1
            elif ref is not None and tr.surface is None:
                miss += 1
            elif ref is not None:
                errs.append(abs(tr.surface - ref))
                its.append(len(tr.iters))
        e = np.array(errs) if errs else np.array([np.nan])
        return e, miss, false_hit, (np.mean(its) if its else float("nan"))

    for label, kw in (("1 ball", dict(nballs=1, nrays=250)),
                      ("3 balls", dict(nballs=3, nrays=250)),
                      ("8 balls", dict(nballs=8, nrays=250)),
                      ("1-6 balls, angled ray",
                       dict(nballs=(1, 7), nrays=300, angled=True)),
                      ("1-6 balls, origin inside",
                       dict(nballs=(1, 7), nrays=300, inside=True, angled=True))):
        e, miss, false_hit, it = fixed_sweep(**kw)
        ok = bool(np.isfinite(e).any()) and np.nanmax(e) < 1e-6 \
            and miss == 0 and false_hit == 0
        row(f"CORRECTED root vs bisection: {label}", ok,
            f"n={int(np.isfinite(e).sum())} p50={np.nanmedian(e):.1e} "
            f"max={np.nanmax(e):.1e} miss={miss} false={false_hit} "
            f"iters={it:.1f}")

    # 6b-i. THE RADIUS TEST. A lone ball's surface must sit at the analytic
    #       meta-radius, not at the sphere radius. Solved straight from the
    #       field, so it does not depend on the reference tracer at all.
    #       This is the check that catches "it's just hitting the circle".
    for label, fn, expect in (("corrected", C.trace_ray_fixed, True),
                              ("PoC", C.trace_ray, False)):
        worst = 0.0
        worst_case = ""
        as_sphere = 0
        for R in (0.5, 1.0, 1.5, 2.5):
            for T in (0.1, 0.2, 0.4, 0.8):
                want = C.meta_radius(R, T)
                for cx in (2.0, 4.0):
                    tr = fn([[cx, 0.0, R]], [0, 0], [1, 0], T)
                    if tr.surface is None:
                        worst = float("inf")
                        worst_case = f"R={R} T={T} MISS"
                        continue
                    got = abs(cx - tr.surface)
                    if abs(got - want) > worst:
                        worst = abs(got - want)
                        worst_case = f"R={R} T={T} got r={got:.4f} want {want:.4f}"
                    if abs(got - R) < 0.02 * R:
                        as_sphere += 1
        ok = worst < 1e-6
        row(f"lone ball sits at the meta-radius  [{label}]", ok,
            f"max radius error {worst:.2e}  ({worst_case})"
            + (f"  -- hit the SPHERE radius in {as_sphere}/32 cases" if as_sphere else ""),
            expect, "" if expect else "O1+O5")

    # 6b-ii. BLOBBING. Two balls placed so that NEITHER alone reaches the
    #        threshold anywhere on the ray -- a union-of-spheres model must
    #        miss entirely. Only summing the fields produces a surface, so a
    #        hit here proves the densities actually composite.
    R, T = 1.5, 0.2
    # separation chosen so each ball contributes ~0.153 at the midline
    s = 0.744 * R
    sph = [[3.0, s, R], [3.0, -s, R]]
    lone_hit = C.trace_ray_fixed([[3.0, s, R]], [0, 0], [1, 0], T).surface
    both = C.trace_ray_fixed(sph, [0, 0], [1, 0], T)
    ref, _, _ = C.reference_hit(sph, [0, 0], [1, 0], T)
    peak = float(C.field(np.array([3.0, 0.0]), sph))
    single_peak = float(C.field(np.array([3.0, 0.0]), [[3.0, s, R]]))
    ok = (lone_hit is None and both.surface is not None
          and ref is not None and abs(both.surface - ref) < 1e-6)
    row("blobbing: sum of two sub-threshold balls", ok,
        f"each alone {single_peak:.3f} < T, together {peak:.3f} >= T; "
        f"lone ray {'misses' if lone_hit is None else 'HITS'}, "
        f"pair hits {both.surface if both.surface is None else round(both.surface, 6)} "
        f"(ref {ref if ref is None else round(ref, 6)})")

    # 6b-iii. The neck: sweep two balls apart and check the algorithm agrees
    #         with the reference about WHERE the blend breaks. The midline ray
    #         hits while the fields still sum past the threshold and misses
    #         once they do not; getting that transition in the right place is
    #         the sharpest statement that blending is modelled correctly.
    R, T = 1.2, 0.2
    rm = C.meta_radius(R, T)
    disagree = 0
    worst = 0.0
    merged_upto = None
    first_gap = None
    for gap in np.arange(0.30, 1.30, 0.02) * R:
        sph = [[3.0, gap, R], [3.0, -gap, R]]
        got = C.trace_ray_fixed(sph, [0, 0], [1, 0], T).surface
        ref, _, _ = C.reference_hit(sph, [0, 0], [1, 0], T)
        if (got is None) != (ref is None):
            disagree += 1
        elif got is not None:
            worst = max(worst, abs(got - ref))
            merged_upto = gap
            if first_gap is None:
                first_gap = gap
    # The neck must survive past the point where the lone surfaces separate,
    # otherwise nothing is blending.
    blends = merged_upto is not None and merged_upto > rm
    ok = disagree == 0 and worst < 1e-6 and blends
    row("blobbing: neck matches reference as balls separate", ok,
        f"lone meta-radius {rm:.3f}; neck holds to half-gap {merged_upto:.3f} "
        f"({merged_upto / rm:.2f}x that), transition matches ref "
        f"({disagree} disagreements), max err {worst:.1e}")

    # 6c. Order independence of the corrected implementation.
    bad = 0
    for _ in range(80):
        sph = [[float(rng.uniform(1.5, 5.0)), float(rng.uniform(-1.0, 1.0)),
                float(rng.uniform(0.6, 1.5))] for _ in range(4)]
        a_ = C.trace_ray_fixed(sph, [0, 0], [1, 0], 0.2).surface
        b_ = C.trace_ray_fixed(list(reversed(sph)), [0, 0], [1, 0], 0.2).surface
        if (a_ is None) != (b_ is None) or (a_ is not None and abs(a_ - b_) > 1e-12):
            bad += 1
    row("CORRECTED result independent of input order", bad == 0,
        f"{bad}/80 scenes differ")

    # 7. Presets do not raise.
    bad = []
    for name, p in C.PRESETS.items():
        for fn in (C.trace_ray, C.trace_ray_fixed):
            try:
                fn(p["spheres"], p["origin"],
                   [math.cos(p["angle"]), math.sin(p["angle"])], 0.2)
            except Exception as ex:
                bad.append(f"{name}/{fn.__name__}: {type(ex).__name__}")
    row("all presets trace without raising", not bad,
        "; ".join(bad) or f"{len(C.PRESETS)} presets x2 impls ok")

    # 8. Optional cross-check against the `bezier` package.
    try:
        import bezier as bz
        ctrl = np.array([[0, 0], [1 / 6, 0.2], [2 / 6, 0.7], [0.5, 1.1],
                         [4 / 6, 0.7], [5 / 6, 0.2], [1, 0]], dtype=float)
        ts = np.linspace(0, 1, 33)
        mine = C.eval_bezier(ctrl, ts)
        theirs = bz.Curve.from_nodes(np.asfortranarray(ctrl.T)).evaluate_multi(ts).T
        d = float(np.max(np.abs(mine - theirs)))
        row("de Casteljau vs `bezier` package", d < 1e-12, f"max abs diff {d:.3e}")
    except ImportError:
        row("de Casteljau vs `bezier` package", None, "skipped (package not installed)")

    # ---- report ----
    width = max(len(n) for n, _, _, _, _ in rows)
    rule = "-" * (width + 46)
    print()
    print("  Bezier-clipped metaballs -- self test")
    print("  PASS = correct   KNOWN = documented defect, see Findings")
    print("  FAIL = regression   FIXED = defect gone, update the expectation")
    print("  " + rule)

    npass = nknown = nfail = nfixed = nskip = 0
    for name, ok, detail, expect, ref in rows:
        if ok is None:
            tag = "SKIP"; nskip += 1
        elif ok and expect:
            tag = "PASS"; npass += 1
        elif not ok and not expect:
            tag = "KNOWN"; nknown += 1
        elif ok and not expect:
            tag = "FIXED"; nfixed += 1
        else:
            tag = "FAIL"; nfail += 1
        suffix = f"  ({ref})" if ref and tag == "KNOWN" else ""
        print(f"  [{tag:5s}] {name.ljust(width)}   {detail}{suffix}")

    print("  " + rule)
    parts = [f"{npass} pass", f"{nknown} known"]
    if nfixed:
        parts.append(f"{nfixed} FIXED")
    if nfail:
        parts.append(f"{nfail} FAIL")
    if nskip:
        parts.append(f"{nskip} skipped")
    print("  " + ", ".join(parts))
    if nfixed:
        print("  A known defect now passes -- flip its `expect` in selftest().")
    print()
    return 1 if nfail else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run the numeric checks and exit")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    Lab().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
