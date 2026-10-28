import subprocess
import sys
import pathlib

HERE = pathlib.Path(__file__).parent


def main():
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])

    try:
        import tkinterdnd2
        tkdnd_dir = pathlib.Path(tkinterdnd2.__file__).parent / 'tkdnd'
    except ImportError:
        print("ERROR: tkinterdnd2 not found — activate the frag-analysis conda env first.")
        sys.exit(1)

    tkdnd_data = f"{tkdnd_dir};tkinterdnd2/tkdnd"

    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', 'Fragment Analyzer',
        '--windowed',           # hide the console window
        '--onedir',             # folder bundle
        '--noconfirm',          # overwrite dist/ without prompting
        '--clean',              # start from a clean build cache
        f'--add-data={tkdnd_data}',
        # scipy dynamic imports not always picked up automatically
        '--hidden-import=scipy.special._cython_special',
        '--hidden-import=scipy.linalg.cython_blas',
        '--hidden-import=scipy.linalg.cython_lapack',
        '--collect-submodules=scipy.interpolate',
        '--collect-submodules=scipy.signal',
        str(HERE / 'fragment_analyzer.py'),
    ]

    print("\nBuilding…")
    print("  " + " ".join(cmd[2:]) + "\n")
    subprocess.check_call(cmd, cwd=HERE)

    out_exe = HERE / 'dist' / 'Fragment Analyzer' / 'Fragment Analyzer.exe'
    print(f"\nDone! {out_exe}")


if __name__ == '__main__':
    main()
