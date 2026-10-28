# Fragment Analyzer

A python based app for viewing fragment analysis data. It reads Applied Biosystems `.fsa` (ABIF) files, plots the fluorescence traces, calls fragment sizes from the ladder, picks peaks, and exports the results to CSV.

![Fragment Analyzer GUI](resources/Screenshot%202026-10-28%20085114.png)

## Purpose

Fragment analysis runs (for example genotyping or indel screens on an ABI sequencer) produce one `.fsa` file per sample. Each file holds the raw traces for up to four dye channels plus a size-standard (ladder) channel. This tool lets you:

- open a single file, several files, or a whole folder (drag and drop works too)
- view each sample on its own, or overlay every loaded sample on one dye channel
- convert the x-axis from scan number to base pairs using the ladder
- find peaks automatically, or add and remove them by clicking on the trace
- save customizable plots from the GUI
- export the peak table and the raw signal to CSV for downstream analysis

## Installation

The easiest way to use this tool is by downloading the pre build executable from the releases tab.

### 1. Create the conda environment

You need [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda.

```bash
conda env create -f environment.yml
conda activate frag-analysis
```

This installs Python 3.11, numpy, scipy, matplotlib and Tk from conda-forge, and `tkinterdnd2` (drag and drop) and `pyinstaller` from pip.


### 2. Run from source

```bash
python fragment_analyzer.py
```

### 3. Build a standalone Windows executable

With the `frag-analysis` environment active:

```bash
python build_exe.py
```

The script runs PyInstaller (installing it if it is missing) and writes the app to:

```
dist/Fragment Analyzer/Fragment Analyzer.exe
```

## Usage instructions

1. **Load data.** Click **Open File** or **Open Dir**, press `Ctrl+O` / `Ctrl+D`, or drag files or a folder onto the window. Loaded samples appear in the file list. Ctrl-click or Shift-click to select several.
2. **Choose what to plot.**
   - *View Mode*: **Single File** shows the active sample, with the channels ticked under *Channels (Single View)*. **Overlay All** draws all loaded samples on the channel picked under *Overlay Channel*.
   - Use the toolbar under the plot to zoom and pan.
3. **Call sizes (optional, needed for base-pair values).**
   - Pick the **Size Standard** (for example *GS 400HD ROX* or *GS 600 LIZ*). The fragment sizes fill in below and can be edited, or choose *Custom* to type your own.
   - Pick the **Ladder channel**: `4` or `5` for the dye that carries the ladder, or `Auto`. On the instrument this was developed against, the ladder is on channel 4.
   - Click **Attempt Size Calling**, then tick **X-axis: Base Pairs**.

   The size standard named inside the `.fsa` file can be wrong, so choose the ladder yourself and check that the ladder peaks line up.
4. **Pick peaks.**
   - **Auto-Detect Peaks** uses *Min height* (RFU) and *Min dist (scans)*. It runs on the files currently shown and only on the channels ticked under *Channels (Single View)*.
   - Or **left-click** the trace to add a peak and **right-click** a marker to remove it.
   - Peaks appear in the table on the right (sample, channel, scan, bp, height). Use **Remove Selected** or **Clear All Peaks** to delete them.
5. **Export.**
   - **Export Peaks CSV** writes the peak table: `sample_name, file, channel, dye_name, scan, bp, height_RFU`.
   - **Export Signal CSV** writes the full traces for the *selected* files into one CSV: `sample_name, file, scan, bp, <dye channels>, LIZ`. The `bp` column is blank for files that have not been size-called.
6. **Appearance.** **Plot Settings…** changes line width, transparency, channel colours and peak markers.

## Notes and limitations

- Size calling needs a clean ladder. Weak ladder signal, or a very tall sample peak sitting on a ladder peak, can still cause inconsistencies in size calling.
- Auto-detect of the ladder from the file's `StdF` annotation does not always match the ladder actually run, so select it explicitly.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
