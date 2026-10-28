import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import csv
import re

from tkinterdnd2 import TkinterDnD, DND_FILES

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.cm as cm

from fsa_reader import FSAFile, CHANNEL_COLORS, KNOWN_LADDERS


@dataclass
class Peak:
    file_path: str
    sample_name: str
    channel: int
    dye_name: str
    scan: int
    bp: Optional[float]
    height: float


CH_COLORS = CHANNEL_COLORS  # {1: blue, 2: green, 3: orange, 4: red, 5: purple}
CH_LABELS = {1: '6-FAM', 2: 'VIC', 3: 'NED', 4: 'PET', 5: 'LIZ'}

PAD = 4

class FragmentAnalyzerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fragment Analyzer")
        self.root.geometry("1400x820")

        # State
        self.file_paths: list[str] = []          # ordered list preserving load order
        self.files: dict[str, FSAFile] = {}       # path → FSAFile
        self.bp_arrays: dict[str, np.ndarray] = {}
        self.active_path: Optional[str] = None
        self.peaks: list[Peak] = []
        self.use_bp = False

        # Per-channel visibility
        self.ch_visible: dict[int, tk.BooleanVar] = {
            i: tk.BooleanVar(value=(i == 1)) for i in range(1, 6)
        }

        # Overlay channel selection
        self.overlay_ch = tk.IntVar(value=1)

        # View mode
        self.view_mode = tk.StringVar(value='single')

        # Peak markers drawn on the plot
        self._peak_artists: list = []

        # Size standard ladder selection
        self.ladder_var = tk.StringVar(value='Auto-detect from file')
        # Channel carrying the size standard: 'Auto', '4' or '5'
        self.ladder_channel_var = tk.StringVar(value='4')

        # Multi-file selection
        self.selected_paths: list[str] = []
        self._settings_win: Optional[tk.Toplevel] = None

        # Visual settings 
        self.line_width = tk.DoubleVar(value=0.9)
        self.line_alpha = tk.DoubleVar(value=0.95)
        # Per-channel colors
        self.ch_colors: dict[int, tk.StringVar] = {
            i: tk.StringVar(value=CH_COLORS[i]) for i in range(1, 6)
        }
        self.peak_marker      = tk.StringVar(value='v')
        self.peak_marker_size = tk.DoubleVar(value=9.0)
        self.peak_color_mode  = tk.StringVar(value='match')   # 'match' | 'custom'
        self.peak_custom_color = tk.StringVar(value='#333333')


        self.xaxis_auto = tk.BooleanVar(value=True)
        self.xaxis_min  = tk.StringVar(value='')
        self.xaxis_max  = tk.StringVar(value='')
        self.xaxis_tick = tk.StringVar(value='')
        self.yaxis_auto = tk.BooleanVar(value=True)
        self.yaxis_min  = tk.StringVar(value='')
        self.yaxis_max  = tk.StringVar(value='')
        self.yaxis_tick = tk.StringVar(value='')

        self._build_ui()
        # Live-redraw when any visual setting changes
        for var in [self.line_width, self.line_alpha, self.peak_marker,
                    self.peak_marker_size, self.peak_color_mode, self.peak_custom_color,
                    self.xaxis_auto, self.yaxis_auto,
                    *self.ch_colors.values()]:
            var.trace_add('write', lambda *_: self._refresh_plot())
        self._refresh_plot()


    def _build_ui(self):
        self._build_menu()

        outer = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=PAD, pady=PAD)

        self._build_left_panel(outer)
        self._build_center_panel(outer)
        self._build_right_panel(outer)

        self.status_var = tk.StringVar(value="Ready — open a file or directory to begin, or drag & drop.")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W, padding=(6, 2)
                  ).pack(side=tk.BOTTOM, fill=tk.X)

        self._register_drop_targets()

    def _build_menu(self):
        bar = tk.Menu(self.root)

        file_menu = tk.Menu(bar, tearoff=0)
        file_menu.add_command(label="Open File…",       command=self.open_file,      accelerator="Ctrl+O")
        file_menu.add_command(label="Open Directory…",  command=self.open_directory, accelerator="Ctrl+D")
        file_menu.add_separator()
        file_menu.add_command(label="Export Peaks CSV…",  command=self.export_peaks)
        file_menu.add_command(label="Export Signal CSV…", command=self.export_signal)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit)
        bar.add_cascade(label="File", menu=file_menu)

        self.root.config(menu=bar)
        self.root.bind('<Control-o>', lambda _: self.open_file())
        self.root.bind('<Control-d>', lambda _: self.open_directory())

    def _build_left_panel(self, parent):
        left = ttk.Frame(parent, width=230)
        parent.add(left, weight=0)

        ttk.Label(left, text="Files", font=('TkDefaultFont', 10, 'bold')).pack(pady=(4, 2))
        ttk.Label(left, text="drag & drop files or a folder here",
                  foreground='gray', font=('TkDefaultFont', 8)).pack()
        bf = ttk.Frame(left)
        bf.pack(fill=tk.X, padx=PAD)
        ttk.Button(bf, text="Open File",  command=self.open_file).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(bf, text="Open Dir",   command=self.open_directory).pack(side=tk.LEFT, expand=True, fill=tk.X)

        lf = ttk.Frame(left)
        lf.pack(fill=tk.BOTH, expand=True, padx=PAD, pady=4)
        vsb = ttk.Scrollbar(lf, orient=tk.VERTICAL)
        self.file_lb = tk.Listbox(lf, selectmode=tk.EXTENDED, exportselection=False,
                                   yscrollcommand=vsb.set, activestyle='dotbox')
        vsb.config(command=self.file_lb.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_lb.pack(fill=tk.BOTH, expand=True)
        self.file_lb.bind('<<ListboxSelect>>', self._on_file_select)
        self.file_lb.bind('<ButtonRelease-1>', self._on_lb_click)

        sep = ttk.Separator(left, orient=tk.HORIZONTAL)

        sep.pack(fill=tk.X, padx=PAD, pady=4)
        ttk.Label(left, text="View Mode").pack()
        for label, val in [("Single File", "single"), ("Overlay All", "overlay")]:
            ttk.Radiobutton(left, text=label, variable=self.view_mode,
                            value=val, command=self._refresh_plot).pack(anchor=tk.W, padx=12)

        ttk.Label(left, text="Overlay Channel").pack(pady=(4, 0))
        for ch in range(1, 5):
            ttk.Radiobutton(left, text=f"Ch{ch}  {CH_LABELS[ch]}",
                            variable=self.overlay_ch, value=ch,
                            command=self._refresh_plot).pack(anchor=tk.W, padx=12)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=PAD, pady=4)
        ttk.Label(left, text="Channels (Single View)").pack()
        for ch in range(1, 5):
            ttk.Checkbutton(left, text=f"Ch{ch}  {CH_LABELS[ch]}",
                            variable=self.ch_visible[ch],
                            command=self._refresh_plot).pack(anchor=tk.W, padx=12)
        ttk.Checkbutton(left, text="Ch5  LIZ",
                        variable=self.ch_visible[5],
                        command=self._refresh_plot).pack(anchor=tk.W, padx=12)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=PAD, pady=4)
        ttk.Label(left, text="Size Standard").pack()

        ladder_names = list(KNOWN_LADDERS.keys())
        self.ladder_cb = ttk.Combobox(left, textvariable=self.ladder_var,
                                       values=ladder_names, state='readonly', width=26)
        self.ladder_cb.pack(fill=tk.X, padx=PAD, pady=(2, 0))
        self.ladder_cb.bind('<<ComboboxSelected>>', self._on_ladder_select)

        ch_row = ttk.Frame(left)
        ch_row.pack(fill=tk.X, padx=PAD, pady=(4, 0))
        ttk.Label(ch_row, text="Ladder channel:").pack(side=tk.LEFT)
        ttk.Combobox(ch_row, textvariable=self.ladder_channel_var,
                     values=['Auto', '4', '5'], state='readonly',
                     width=6).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(left, text="Fragment sizes (bp, comma-separated):").pack(anchor=tk.W, padx=PAD, pady=(4, 0))
        self.ladder_sizes_var = tk.StringVar()
        sizes_entry = ttk.Entry(left, textvariable=self.ladder_sizes_var, width=28)
        sizes_entry.pack(fill=tk.X, padx=PAD, pady=(0, 2))

        ttk.Button(left, text="Attempt Size Calling",
                   command=self._do_size_calling).pack(fill=tk.X, padx=PAD, pady=2)
        self.use_bp_var = tk.BooleanVar(value=False)
        self.bp_cb = ttk.Checkbutton(left, text="X-axis: Base Pairs",
                                      variable=self.use_bp_var,
                                      command=self._toggle_bp,
                                      state=tk.DISABLED)
        self.bp_cb.pack(anchor=tk.W, padx=12)

        # Peak detection
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=PAD, pady=4)
        ttk.Label(left, text="Peak Detection").pack()

        tf = ttk.Frame(left)
        tf.pack(fill=tk.X, padx=PAD)
        ttk.Label(tf, text="Min height:").pack(side=tk.LEFT)
        self.peak_thresh = tk.DoubleVar(value=200)
        ttk.Entry(tf, textvariable=self.peak_thresh, width=8).pack(side=tk.RIGHT)

        mf = ttk.Frame(left)
        mf.pack(fill=tk.X, padx=PAD)
        ttk.Label(mf, text="Min dist (scans):").pack(side=tk.LEFT)
        self.peak_dist = tk.IntVar(value=10)
        ttk.Entry(mf, textvariable=self.peak_dist, width=8).pack(side=tk.RIGHT)

        ttk.Button(left, text="Auto-Detect Peaks",
                   command=self._auto_detect_peaks).pack(fill=tk.X, padx=PAD, pady=2)
        ttk.Button(left, text="Clear All Peaks",
                   command=self._clear_peaks).pack(fill=tk.X, padx=PAD, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=PAD, pady=4)
        ttk.Button(left, text="Export Peaks CSV",
                   command=self.export_peaks).pack(fill=tk.X, padx=PAD, pady=2)
        ttk.Button(left, text="Export Signal CSV",
                   command=self.export_signal).pack(fill=tk.X, padx=PAD, pady=2)
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=PAD, pady=4)
        ttk.Button(left, text="Plot Settings...",
                   command=self._open_settings_dialog).pack(fill=tk.X, padx=PAD, pady=2)

    def _build_center_panel(self, parent):
        center = ttk.Frame(parent)
        parent.add(center, weight=3)

        ttk.Label(center,
                  text="Left-click to add a peak  ·  Right-click an existing marker to remove it",
                  foreground='gray').pack(side=tk.TOP, fill=tk.X, padx=PAD)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_frame = ttk.Frame(center)
        tb_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb_frame)
        self.toolbar.update()

        self.canvas.mpl_connect('button_press_event', self._on_plot_click)

    def _build_right_panel(self, parent):
        right = ttk.Frame(parent, width=310)
        parent.add(right, weight=1)

        ttk.Label(right, text="Peaks", font=('TkDefaultFont', 10, 'bold')).pack(pady=(4, 2))

        cols = ('sample', 'ch', 'scan', 'bp', 'height')
        self.peak_tree = ttk.Treeview(right, columns=cols, show='headings', selectmode='extended')
        for col, width, label in [
            ('sample', 90,  'Sample'),
            ('ch',     45,  'Ch'),
            ('scan',   60,  'Scan'),
            ('bp',     55,  'bp'),
            ('height', 65,  'Height'),
        ]:
            self.peak_tree.heading(col, text=label,
                                   command=lambda c=col: self._sort_peaks(c))
            self.peak_tree.column(col, width=width, anchor=tk.CENTER)

        vsb = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.peak_tree.yview)
        self.peak_tree.config(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.peak_tree.pack(fill=tk.BOTH, expand=True, padx=PAD)

        ttk.Button(right, text="Remove Selected",
                   command=self._remove_selected_peaks).pack(fill=tk.X, padx=PAD, pady=4)

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open FSA File",
            filetypes=[("FSA files", "*.fsa"), ("All files", "*.*")]
        )
        if path:
            self._load_file(path)

    def open_directory(self):
        d = filedialog.askdirectory(title="Open Directory of FSA Files")
        if not d:
            return
        paths = sorted(Path(d).rglob("*.fsa"))
        if not paths:
            messagebox.showwarning("No Files", "No .fsa files found in that directory.")
            return
        for p in paths:
            self._load_file(str(p))
        self.status_var.set(f"Loaded {len(paths)} file(s) from {Path(d).name}/")

    def _load_file(self, path: str):
        if path in self.files:
            return
        try:
            fsa = FSAFile(path)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not read:\n{path}\n\n{exc}")
            return

        self.files[path] = fsa
        self.file_paths.append(path)
        self.file_lb.insert(tk.END, fsa.sample_name or fsa.name)

        if self.active_path is None:
            self.active_path = path
            self.file_lb.selection_set(0)
            self.selected_paths = [path]

        self._refresh_plot()
        n = len(self.file_paths)
        self.status_var.set(f"{n} file(s) loaded — active: {fsa.sample_name or fsa.name}")

    def _on_file_select(self, _=None):
        sel = self.file_lb.curselection()
        self.selected_paths = [self.file_paths[i] for i in sel if i < len(self.file_paths)]
        if self.selected_paths and self.active_path not in self.selected_paths:
            self.active_path = self.selected_paths[-1]
        self._refresh_plot()
        n = len(self.selected_paths)
        if n == 1:
            self.status_var.set(f"Active: {self.files[self.active_path].sample_name}")
        elif n > 1:
            self.status_var.set(f"{n} files selected — active: {self.files[self.active_path].sample_name}")

    def _on_lb_click(self, event):
        """Track which item was directly clicked so active_path stays sensible."""
        idx = self.file_lb.nearest(event.y)
        if 0 <= idx < len(self.file_paths):
            self.active_path = self.file_paths[idx]

    def _register_drop_targets(self):
        """Register DnD_FILES drop on the file list and the plot canvas."""
        for widget in (self.file_lb, self.canvas.get_tk_widget()):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind('<<Drop>>', self._on_drop)

    def _on_drop(self, event):
        """Handle a drag-and-drop event carrying one or more paths."""
        paths = self._parse_drop_paths(event.data)
        loaded = 0
        for p in paths:
            path = Path(p)
            if path.is_dir():
                fsa_files = sorted(path.rglob('*.fsa'))
                for fp in fsa_files:
                    self._load_file(str(fp))
                    loaded += 1
            elif path.suffix.lower() == '.fsa':
                self._load_file(str(path))
                loaded += 1
        if loaded:
            self.status_var.set(f"Dropped {loaded} file(s) loaded.")

    @staticmethod
    def _parse_drop_paths(data: str) -> list[str]:
        """Parse the tkdnd data string into a list of file paths.

        tkdnd encodes paths as space-separated tokens; paths containing
        spaces are wrapped in curly braces: {C:/path with spaces/f.fsa}.
        """
        paths = []
        remaining = data
        for match in re.finditer(r'\{([^}]+)\}', data):
            paths.append(match.group(1))
            remaining = remaining.replace(match.group(0), '')
        paths.extend(tok for tok in remaining.split() if tok)
        return paths

    def _xdata(self, path: str):
        """Return (x_array, x_label) for the given file."""
        fsa = self.files[path]
        if self.use_bp and path in self.bp_arrays:
            return self.bp_arrays[path], 'Size (bp)'
        return fsa.scan_numbers, 'Scan Number'

    def _scan_from_xval(self, path: str, xval: float) -> int:
        """Return scan index nearest to an x-axis value."""
        x, _ = self._xdata(path)
        return int(np.argmin(np.abs(x - xval)))

    def _display_paths(self) -> list[str]:
        """Files currently shown in the plot (mirrors _refresh_plot dispatch logic)."""
        selected = self.selected_paths or ([self.active_path] if self.active_path else [])
        if self.view_mode.get() == 'overlay':
            return selected or self.file_paths
        return selected

    def _apply_axis_limits(self):
        """Apply user-specified axis limits and tick spacing to the current axes."""
        from matplotlib.ticker import MultipleLocator, AutoLocator

        def _f(s):
            try:
                return float(s.strip())
            except (ValueError, AttributeError):
                return None

        if not self.xaxis_auto.get():
            xmin = _f(self.xaxis_min.get())
            xmax = _f(self.xaxis_max.get())
            cur  = self.ax.get_xlim()
            self.ax.set_xlim(
                xmin if xmin is not None else cur[0],
                xmax if xmax is not None else cur[1],
            )
        xtick = _f(self.xaxis_tick.get())
        if xtick and xtick > 0:
            self.ax.xaxis.set_major_locator(MultipleLocator(xtick))
        elif not self.xaxis_tick.get().strip():
            self.ax.xaxis.set_major_locator(AutoLocator())

        if not self.yaxis_auto.get():
            ymin = _f(self.yaxis_min.get())
            ymax = _f(self.yaxis_max.get())
            cur  = self.ax.get_ylim()
            self.ax.set_ylim(
                ymin if ymin is not None else cur[0],
                ymax if ymax is not None else cur[1],
            )
        ytick = _f(self.yaxis_tick.get())
        if ytick and ytick > 0:
            self.ax.yaxis.set_major_locator(MultipleLocator(ytick))
        elif not self.yaxis_tick.get().strip():
            self.ax.yaxis.set_major_locator(AutoLocator())

    def _refresh_plot(self):
        self.ax.cla()
        self._peak_artists.clear()

        if not self.files:
            self.ax.text(0.5, 0.5, 'Open an .fsa file or directory to begin',
                         ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=12, color='gray')
            self.canvas.draw_idle()
            return

        display = self._display_paths()

        if self.view_mode.get() == 'overlay':
            self._plot_overlay(display)
        else:
            if len(display) > 1:
                self._plot_multi(display)
            elif display:
                self._plot_single(display[0])

        self._draw_peaks()
        self._apply_axis_limits()
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw_idle()

    def _plot_single(self, path: str):
        fsa = self.files[path]
        x, xlabel = self._xdata(path)
        lw    = self.line_width.get()
        alpha = self.line_alpha.get()

        for ch in range(1, 5):
            if not self.ch_visible[ch].get():
                continue
            data = fsa.get_channel_data(ch)
            if data is None:
                continue
            # Matrix-corrected channels can have negative values; clipping to 0
            # prevents a phantom "floor line" when set_ylim(bottom=0) is active.
            self.ax.plot(x, np.maximum(data, 0), color=self.ch_colors[ch].get(),
                         linewidth=lw, label=fsa.get_channel_name(ch), alpha=alpha)

        if self.ch_visible[5].get() and fsa.liz_channel is not None:
            self.ax.plot(x, fsa.liz_channel, color=self.ch_colors[5].get(),
                         linewidth=max(lw * 0.8, 0.5), label='LIZ',
                         alpha=alpha * 0.75, linestyle='--')

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel('Fluorescence (RFU)')
        self.ax.set_title(fsa.sample_name or fsa.name)
        if self.ax.get_lines():
            self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True, alpha=0.25, linestyle=':')
        self.ax.set_ylim(bottom=0)

    def _plot_multi(self, paths: list):
        """Show multiple selected files overlaid on the overlay channel."""
        ch      = self.overlay_ch.get()
        lw      = self.line_width.get()
        alpha   = self.line_alpha.get()
        palette = cm.tab20(np.linspace(0, 1, max(len(paths), 1)))

        for path, color in zip(paths, palette):
            fsa  = self.files[path]
            data = fsa.get_channel_data(ch)
            if data is None:
                continue
            x, xlabel = self._xdata(path)
            is_active  = (path == self.active_path)
            self.ax.plot(x, np.maximum(data, 0), color=color,
                         linewidth=lw * (1.4 if is_active else 0.75),
                         alpha=alpha if is_active else alpha * 0.6,
                         label=fsa.sample_name or fsa.name)

        ch_name = CH_LABELS.get(ch, f'Ch{ch}')
        x_lbl   = 'Size (bp)' if self.use_bp else 'Scan Number'
        self.ax.set_xlabel(x_lbl)
        self.ax.set_ylabel('Fluorescence (RFU)')
        self.ax.set_title(f'Selected files — {ch_name}')
        self.ax.grid(True, alpha=0.25, linestyle=':')
        self.ax.set_ylim(bottom=0)
        if len(paths) <= 16:
            self.ax.legend(loc='upper right', fontsize=7)

    def _plot_overlay(self, paths: list):
        ch      = self.overlay_ch.get()
        lw      = self.line_width.get()
        alpha   = self.line_alpha.get()
        palette = cm.tab20(np.linspace(0, 1, max(len(paths), 1)))

        for path, color in zip(paths, palette):
            fsa  = self.files[path]
            data = fsa.get_channel_data(ch)
            if data is None:
                continue
            x, _ = self._xdata(path)
            is_active = (path == self.active_path)
            self.ax.plot(x, np.maximum(data, 0), color=color,
                         linewidth=lw * (1.4 if is_active else 0.75),
                         alpha=alpha if is_active else alpha * 0.6,
                         label=fsa.sample_name or fsa.name)

        ch_name = CH_LABELS.get(ch, f'Ch{ch}')
        x_lbl   = 'Size (bp)' if self.use_bp else 'Scan Number'
        self.ax.set_xlabel(x_lbl)
        self.ax.set_ylabel('Fluorescence (RFU)')
        self.ax.set_title(f'Overlay — {ch_name}')
        self.ax.grid(True, alpha=0.25, linestyle=':')
        self.ax.set_ylim(bottom=0)
        if len(paths) <= 16:
            self.ax.legend(loc='upper right', fontsize=7)

    def _draw_peaks(self):
        self._peak_artists.clear()
        is_overlay   = self.view_mode.get() == 'overlay'
        multi_select = len(self.selected_paths) > 1
        active_ch    = self.overlay_ch.get()
        display_paths = set(self.selected_paths or ([self.active_path] if self.active_path else []))

        mshape = self.peak_marker.get()
        msize  = self.peak_marker_size.get()

        for peak in self.peaks:
            if peak.file_path not in self.files:
                continue
            if peak.file_path not in display_paths:
                continue
            # In overlay / multi-select views only show the plotted channel
            if (is_overlay or multi_select) and peak.channel != active_ch:
                continue

            x, _ = self._xdata(peak.file_path)
            if peak.scan >= len(x):
                continue
            xv = float(x[peak.scan])
            yv = peak.height

            ch_color = self.ch_colors.get(peak.channel, tk.StringVar(value='#555555')).get()
            mcolor   = ch_color if self.peak_color_mode.get() == 'match' else self.peak_custom_color.get()

            line = self.ax.axvline(x=xv, color=ch_color, linestyle=':', alpha=0.45, linewidth=1)
            marker, = self.ax.plot(xv, yv, marker=mshape, color=mcolor,
                                   markersize=msize, markeredgecolor='black',
                                   markeredgewidth=0.6, zorder=5, picker=6)
            marker._peak_ref = peak
            self._peak_artists.extend([line, marker])

    def _on_plot_click(self, event):
        if event.inaxes != self.ax:
            return
        if self.toolbar.mode:            # zoom / pan active don't interfere
            return
        if event.xdata is None or event.ydata is None:
            return
        if not self.active_path:
            return

        if event.button == 3:            # right-click remove nearest peak
            self._try_remove_peak_at(event.xdata, event.ydata)
            return

        if event.button != 1:
            return

        # Left-click add peak at nearest scan on nearest visible channel
        fsa = self.files[self.active_path]
        scan = self._scan_from_xval(self.active_path, event.xdata)

        if self.view_mode.get() == 'overlay':
            channels_to_check = [self.overlay_ch.get()]
        else:
            channels_to_check = [ch for ch in range(1, 5) if self.ch_visible[ch].get()]

        if not channels_to_check:
            return

        # Find channel whose value at this scan is closest to click y
        best_ch, best_dist = None, float('inf')
        for ch in channels_to_check:
            data = fsa.get_channel_data(ch)
            if data is None or scan >= len(data):
                continue
            dist = abs(float(data[scan]) - event.ydata)
            if dist < best_dist:
                best_dist, best_ch = dist, ch

        if best_ch is None:
            return

        data = fsa.get_channel_data(best_ch)
        height = float(data[scan])
        x, _ = self._xdata(self.active_path)
        bp = float(self.bp_arrays[self.active_path][scan]) if self.active_path in self.bp_arrays else None

        peak = Peak(
            file_path=self.active_path,
            sample_name=fsa.sample_name or fsa.name,
            channel=best_ch,
            dye_name=fsa.get_channel_name(best_ch),
            scan=scan,
            bp=bp,
            height=height,
        )
        self.peaks.append(peak)
        self._sync_peak_table()
        self._refresh_plot()

        bp_txt = f"  {bp:.1f} bp" if bp is not None else ""
        self.status_var.set(
            f"Peak added — {peak.sample_name}  Ch{best_ch}  scan {scan}{bp_txt}  "
            f"height {height:.0f}"
        )

    def _try_remove_peak_at(self, xval, yval):
        """Remove the peak whose marker is closest to the right-click location."""
        if not self.peaks:
            return
        is_overlay   = self.view_mode.get() == 'overlay'
        multi_select = len(self.selected_paths) > 1
        active_ch    = self.overlay_ch.get()
        display_paths = set(self.selected_paths or ([self.active_path] if self.active_path else []))

        best_idx, best_dist = None, float('inf')
        for i, peak in enumerate(self.peaks):
            if peak.file_path not in display_paths:
                continue
            if (is_overlay or multi_select) and peak.channel != active_ch:
                continue
            if peak.file_path not in self.files:
                continue
            x, _ = self._xdata(peak.file_path)
            if peak.scan >= len(x):
                continue
            dist = abs(float(x[peak.scan]) - xval)
            if dist < best_dist:
                best_dist, best_idx = dist, i

        if best_idx is not None and best_dist < (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.02:
            removed = self.peaks.pop(best_idx)
            self._sync_peak_table()
            self._refresh_plot()
            self.status_var.set(f"Peak removed — {removed.sample_name} Ch{removed.channel} scan {removed.scan}")

    def _open_settings_dialog(self):
        """Open (or raise) the non-modal Plot Settings window."""
        if self._settings_win and self._settings_win.winfo_exists():
            self._settings_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Plot Settings")
        win.resizable(False, False)
        self._settings_win = win

        MARKERS = [
            ('Triangle down', 'v'), ('Triangle up', '^'), ('Circle', 'o'),
            ('Square', 's'), ('Diamond', 'D'), ('Star', '*'),
            ('Plus', '+'), ('Cross', 'x'), ('Pentagon', 'p'),
        ]

        def section(text):
            ttk.Separator(win, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=(8, 2))
            ttk.Label(win, text=text, font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, padx=10)

        def row(parent, label, widget_factory):
            f = ttk.Frame(parent)
            f.pack(fill=tk.X, padx=10, pady=2)
            ttk.Label(f, text=label, width=16, anchor=tk.W).pack(side=tk.LEFT)
            widget_factory(f).pack(side=tk.LEFT, fill=tk.X, expand=True)

        section("Line Style")
        row(win, "Width",
            lambda p: tk.Scale(p, variable=self.line_width, from_=0.3, to=4.0,
                               resolution=0.1, orient=tk.HORIZONTAL, length=160))
        row(win, "Opacity",
            lambda p: tk.Scale(p, variable=self.line_alpha, from_=0.1, to=1.0,
                               resolution=0.05, orient=tk.HORIZONTAL, length=160))

        section("Channel Colors")
        ch_labels = {1: '6-FAM', 2: 'VIC', 3: 'NED', 4: 'PET', 5: 'LIZ'}
        for ch in range(1, 6):
            f = ttk.Frame(win)
            f.pack(fill=tk.X, padx=10, pady=2)
            ttk.Label(f, text=f"Ch{ch}  {ch_labels[ch]}", width=16, anchor=tk.W).pack(side=tk.LEFT)
            swatch = tk.Label(f, width=4, relief=tk.SUNKEN,
                              bg=self.ch_colors[ch].get())
            swatch.pack(side=tk.LEFT, padx=(0, 6))

            def _pick(ch=ch, sw=swatch):
                from tkinter.colorchooser import askcolor
                result = askcolor(color=self.ch_colors[ch].get(),
                                  title=f"Color for {ch_labels[ch]}", parent=win)
                if result[1]:
                    self.ch_colors[ch].set(result[1])
                    sw.config(bg=result[1])

            ttk.Button(f, text="Change...", command=_pick).pack(side=tk.LEFT)

        section("Peak Markers")

        # Shape combobox
        f = ttk.Frame(win)
        f.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(f, text="Shape", width=16, anchor=tk.W).pack(side=tk.LEFT)
        shape_display = tk.StringVar(value=next(
            (lbl for lbl, val in MARKERS if val == self.peak_marker.get()), 'Triangle down'))

        def _on_shape_select(_=None):
            chosen = shape_display.get()
            for lbl, val in MARKERS:
                if lbl == chosen:
                    self.peak_marker.set(val)
                    break

        cb = ttk.Combobox(f, textvariable=shape_display,
                          values=[lbl for lbl, _ in MARKERS], state='readonly', width=16)
        cb.pack(side=tk.LEFT)
        cb.bind('<<ComboboxSelected>>', _on_shape_select)

        row(win, "Size",
            lambda p: tk.Scale(p, variable=self.peak_marker_size, from_=3, to=24,
                               resolution=0.5, orient=tk.HORIZONTAL, length=160))

        f = ttk.Frame(win)
        f.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(f, text="Color", width=16, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Radiobutton(f, text="Match channel", variable=self.peak_color_mode,
                        value='match').pack(side=tk.LEFT)
        ttk.Radiobutton(f, text="Custom", variable=self.peak_color_mode,
                        value='custom').pack(side=tk.LEFT, padx=(8, 0))

        f2 = ttk.Frame(win)
        f2.pack(fill=tk.X, padx=10, pady=(0, 2))
        ttk.Label(f2, text="", width=16).pack(side=tk.LEFT)
        custom_swatch = tk.Label(f2, width=4, relief=tk.SUNKEN,
                                  bg=self.peak_custom_color.get())
        custom_swatch.pack(side=tk.LEFT, padx=(0, 6))

        def _pick_peak_color():
            from tkinter.colorchooser import askcolor
            result = askcolor(color=self.peak_custom_color.get(),
                              title="Custom peak color", parent=win)
            if result[1]:
                self.peak_custom_color.set(result[1])
                custom_swatch.config(bg=result[1])

        ttk.Button(f2, text="Change...", command=_pick_peak_color).pack(side=tk.LEFT)

        section("Axis Limits")

        axis_trace_cleanups = []

        def _axis_row(parent, label, auto_var, min_var, max_var, tick_var):
            """Build one axis row: label | Auto cb | Min | Max | Tick."""
            f = ttk.Frame(parent)
            f.pack(fill=tk.X, padx=10, pady=3)
            ttk.Label(f, text=label, width=3, anchor=tk.W).pack(side=tk.LEFT)

            entries = []

            def _toggle(*_):
                state = tk.DISABLED if auto_var.get() else tk.NORMAL
                for e in entries:
                    if e.winfo_exists():
                        e.config(state=state)

            trace_id = auto_var.trace_add('write', _toggle)
            axis_trace_cleanups.append(lambda v=auto_var, tid=trace_id: v.trace_remove('write', tid))

            ttk.Checkbutton(f, text="Auto", variable=auto_var).pack(side=tk.LEFT, padx=(0, 6))

            for lbl, var in [("Min", min_var), ("Max", max_var), ("Tick", tick_var)]:
                ttk.Label(f, text=lbl).pack(side=tk.LEFT)
                e = ttk.Entry(f, textvariable=var, width=7)
                e.pack(side=tk.LEFT, padx=(2, 6))
                e.bind('<Return>',   lambda _: self._refresh_plot())
                e.bind('<FocusOut>', lambda _: self._refresh_plot())
                entries.append(e)

            _toggle()

        _axis_row(win, "X", self.xaxis_auto,
                  self.xaxis_min, self.xaxis_max, self.xaxis_tick)
        _axis_row(win, "Y", self.yaxis_auto,
                  self.yaxis_min, self.yaxis_max, self.yaxis_tick)

        def _on_win_destroy(event):
            if event.widget is win:
                for cleanup in axis_trace_cleanups:
                    try:
                        cleanup()
                    except Exception:
                        pass
        win.bind('<Destroy>', _on_win_destroy)

        ttk.Separator(win, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=(10, 4))
        footer = ttk.Frame(win)
        footer.pack(fill=tk.X, padx=10, pady=(0, 8))

        def _reset():
            self.line_width.set(0.9)
            self.line_alpha.set(0.95)
            for i, default in CH_COLORS.items():
                self.ch_colors[i].set(default)
            self.peak_marker.set('v')
            self.peak_marker_size.set(9.0)
            self.peak_color_mode.set('match')
            self.peak_custom_color.set('#333333')
            self.xaxis_auto.set(True)
            self.xaxis_min.set('')
            self.xaxis_max.set('')
            self.xaxis_tick.set('')
            self.yaxis_auto.set(True)
            self.yaxis_min.set('')
            self.yaxis_max.set('')
            self.yaxis_tick.set('')

        ttk.Button(footer, text="Reset Defaults", command=_reset).pack(side=tk.LEFT)
        ttk.Button(footer, text="Close", command=win.destroy).pack(side=tk.RIGHT)

    def _auto_detect_peaks(self):
        target_paths = self._display_paths()
        if not target_paths:
            messagebox.showinfo("No File", "Select a file first.")
            return

        from scipy.signal import find_peaks as sp_find_peaks

        min_h = self.peak_thresh.get()
        min_d = max(1, self.peak_dist.get())
        channels_to_scan = [ch for ch in range(1, 5) if self.ch_visible[ch].get()]

        # Remove existing auto-detected peaks for all target files × channels
        target_set = set(target_paths)
        self.peaks = [
            p for p in self.peaks
            if not (p.file_path in target_set and p.channel in channels_to_scan)
        ]

        added = 0
        for path in target_paths:
            fsa = self.files[path]
            for ch in channels_to_scan:
                data = fsa.get_channel_data(ch)
                if data is None:
                    continue
                idxs, _ = sp_find_peaks(data, height=min_h, distance=min_d,
                                        prominence=min_h * 0.25)
                for scan in idxs:
                    bp = float(self.bp_arrays[path][scan]) if path in self.bp_arrays else None
                    self.peaks.append(Peak(
                        file_path=path,
                        sample_name=fsa.sample_name or fsa.name,
                        channel=ch,
                        dye_name=fsa.get_channel_name(ch),
                        scan=int(scan),
                        bp=bp,
                        height=float(data[scan]),
                    ))
                    added += 1

        self._sync_peak_table()
        self._refresh_plot()
        n_files = len(target_paths)
        file_desc = f"{n_files} file(s)" if n_files > 1 else self.files[target_paths[0]].sample_name
        self.status_var.set(f"Auto-detected {added} peak(s) across {file_desc}.")

    def _clear_peaks(self):
        self.peaks.clear()
        self._sync_peak_table()
        self._refresh_plot()
        self.status_var.set("All peaks cleared.")

    def _on_ladder_select(self, _=None):
        """Auto-fill the sizes entry when a preset ladder is chosen."""
        name = self.ladder_var.get()
        sizes = KNOWN_LADDERS.get(name)
        if sizes is not None:
            self.ladder_sizes_var.set(', '.join(str(s) for s in sizes))
        elif name == 'Custom':
            # Leave whatever is already in the field so the user can edit it
            pass
        else:
            self.ladder_sizes_var.set('')

    def _parse_ladder_sizes(self):
        """Return a list[int] from the sizes entry, or None for auto-detect."""
        raw = self.ladder_sizes_var.get().strip()
        if not raw:
            return None  # auto-detect
        try:
            sizes = [int(x.strip()) for x in raw.replace(';', ',').split(',') if x.strip()]
            if len(sizes) < 4:
                raise ValueError("Need at least 4 sizes for interpolation.")
            return sorted(sizes)
        except ValueError as exc:
            messagebox.showerror("Invalid Sizes", f"Could not parse fragment sizes:\n{exc}")
            return False

    def _do_size_calling(self):
        if not self.files:
            return

        std_sizes = self._parse_ladder_sizes()
        if std_sizes is False:   # parse error user was already warned
            return

        ch_sel = self.ladder_channel_var.get()
        ladder_ch = int(ch_sel) if ch_sel in ('4', '5') else None

        ok, fail = 0, 0
        for path, fsa in self.files.items():
            bp = fsa.attempt_size_calling(std_sizes=std_sizes,
                                          ladder_channel=ladder_ch)
            if bp is not None:
                self.bp_arrays[path] = bp
                ok += 1
            else:
                fail += 1

        if ok > 0:
            self.bp_cb.config(state=tk.NORMAL)
            msg = f"Size calling succeeded for {ok} file(s)"
            if fail:
                msg += f" (failed for {fail})"
            self.status_var.set(msg + ".")
        else:
            messagebox.showwarning(
                "Size Calling Failed",
                "Could not find enough LIZ peaks to perform size calling.\n"
                "The x-axis will remain in scan numbers."
            )

    def _toggle_bp(self):
        self.use_bp = self.use_bp_var.get()
        # Update bp values on existing peaks
        for peak in self.peaks:
            if peak.file_path in self.bp_arrays:
                peak.bp = float(self.bp_arrays[peak.file_path][peak.scan])
            else:
                peak.bp = None
        self._sync_peak_table()
        self._refresh_plot()

    def _sync_peak_table(self):
        for item in self.peak_tree.get_children():
            self.peak_tree.delete(item)
        for i, p in enumerate(self.peaks):
            bp_str = f"{p.bp:.1f}" if p.bp is not None else "—"
            self.peak_tree.insert('', tk.END, iid=str(i), values=(
                p.sample_name,
                f"Ch{p.channel} {p.dye_name}",
                p.scan,
                bp_str,
                f"{p.height:.0f}",
            ))

    def _sort_peaks(self, col):
        key_map = {'sample': lambda p: p.sample_name,
                   'ch':     lambda p: p.channel,
                   'scan':   lambda p: p.scan,
                   'bp':     lambda p: p.bp or 0,
                   'height': lambda p: p.height}
        if col in key_map:
            self.peaks.sort(key=key_map[col])
            self._sync_peak_table()

    def _remove_selected_peaks(self):
        sel = self.peak_tree.selection()
        if not sel:
            return
        indices = sorted({int(iid) for iid in sel}, reverse=True)
        for idx in indices:
            if idx < len(self.peaks):
                self.peaks.pop(idx)
        self._sync_peak_table()
        self._refresh_plot()

    def export_peaks(self):
        if not self.peaks:
            messagebox.showinfo("No Peaks", "No peaks to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Peaks",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return

        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_name', 'file', 'channel', 'dye_name',
                        'scan', 'bp', 'height_RFU'])
            for p in self.peaks:
                w.writerow([
                    p.sample_name,
                    Path(p.file_path).name,
                    p.channel,
                    p.dye_name,
                    p.scan,
                    f"{p.bp:.2f}" if p.bp is not None else '',
                    f"{p.height:.2f}",
                ])
        self.status_var.set(f"Exported {len(self.peaks)} peak(s) → {Path(path).name}")

    def export_signal(self):
        export_paths = self.selected_paths or (
            [self.active_path] if self.active_path else [])
        export_paths = [p for p in export_paths if p in self.files]
        if not export_paths:
            messagebox.showinfo("No Data", "Select at least one file to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Signal Data",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return

        # One table for all selected files; channel columns are fixed so
        # every row lines up (blank where a file lacks that channel).
        first = self.files[export_paths[0]]
        headers = ['sample_name', 'file', 'scan', 'bp']
        headers += [first.get_channel_name(ch) or f'Ch{ch}' for ch in range(1, 5)]
        headers.append('LIZ')

        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(headers)
            for file_path in export_paths:
                fsa = self.files[file_path]
                fname = Path(file_path).name
                bp = self.bp_arrays.get(file_path)
                chans = [fsa.get_channel_data(ch) for ch in range(1, 5)]
                chans.append(fsa.liz_channel)

                for i in range(fsa.n_scans):
                    row = [fsa.sample_name, fname, i,
                           f"{bp[i]:.2f}" if bp is not None else '']
                    for d in chans:
                        row.append(f"{d[i]:.1f}" if d is not None and i < len(d) else '')
                    w.writerow(row)

        self.status_var.set(
            f"Exported signal data for {len(export_paths)} file(s) → {Path(path).name}")


def main():
    root = TkinterDnD.Tk()
    app = FragmentAnalyzerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
