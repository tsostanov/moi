from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE = SCRIPT_DIR / "native_intersect.c"
OUTPUT = SCRIPT_DIR / ("native_intersect.dll" if sys.platform == "win32" else "native_intersect.so")
WINDOWS_DLL_DIRS = (
    Path(r"C:\msys64\ucrt64\bin"),
    Path(r"C:\msys64\mingw64\bin"),
    Path(r"C:\mingw64\bin"),
)


def python_is_64bit() -> bool:
    return platform.architecture()[0] == "64bit"


def is_gcc_target_compatible(target: str) -> bool:
    normalized = target.strip().lower()
    if sys.platform != "win32":
        return True
    if python_is_64bit():
        return normalized.startswith("x86_64") or normalized.startswith("amd64")
    return normalized.startswith("i686") or normalized.startswith("i386") or normalized.startswith("mingw32")


def gcc_target(compiler: str) -> str | None:
    try:
        result = subprocess.run(
            [compiler, "-dumpmachine"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def iter_gcc_candidates() -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(path: str | None) -> None:
        if not path:
            return
        normalized = str(Path(path))
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

    if sys.platform == "win32":
        add(r"C:\msys64\ucrt64\bin\gcc.exe")
        add(r"C:\msys64\mingw64\bin\gcc.exe")
        add(r"C:\mingw64\bin\gcc.exe")
    add(shutil.which("gcc"))
    return candidates


def select_compiler() -> tuple[str, str, str | None] | None:
    for gcc in iter_gcc_candidates():
        target = gcc_target(gcc)
        if target and is_gcc_target_compatible(target):
            return "gcc", gcc, target

    cl = shutil.which("cl") if sys.platform == "win32" else None
    if cl is not None:
        return "cl", cl, None

    return None


def add_windows_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    for directory in WINDOWS_DLL_DIRS:
        if not directory.exists():
            continue
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(directory))
        except (AttributeError, OSError):
            continue


def main() -> int:
    print("Python:", platform.architecture()[0])
    selection = select_compiler()
    if selection is None:
        print("No supported C compiler was found.")
        if sys.platform == "win32":
            print("Install a 64-bit gcc, or open a Developer Command Prompt with MSVC Build Tools.")
        else:
            print("Install gcc and make sure it is available in PATH.")
        return 1

    compiler_kind, compiler, target = selection
    if compiler_kind == "gcc":
        command = [compiler, "-O3", "-shared"]
        if sys.platform != "win32":
            command.append("-fPIC")
        command.extend(["-o", str(OUTPUT), str(SOURCE)])
        print("Compiler:", compiler)
        if target:
            print("Target:", target)
        print("Command:", " ".join(command))
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0 and not OUTPUT.exists():
            print(f"Compiler exited with code {result.returncode}.")
            return result.returncode
        if result.returncode != 0:
            print(f"warning: compiler exited with code {result.returncode}, but output file was created.")
    else:
        with tempfile.TemporaryDirectory() as build_dir:
            command = [
                compiler,
                "/O2",
                "/LD",
                str(SOURCE),
                "/link",
                f"/OUT:{OUTPUT}",
            ]
            print("Compiler:", compiler)
            print("Command:", " ".join(command))
            subprocess.run(command, check=True, cwd=build_dir)

    try:
        add_windows_dll_dirs()
        ctypes.CDLL(str(OUTPUT))
    except OSError as error:
        print(f"Built {OUTPUT}, but Python could not load it:")
        print(error)
        OUTPUT.unlink(missing_ok=True)
        print(f"Removed incompatible {OUTPUT}")
        print("Most likely cause on Windows: 32-bit compiler with 64-bit Python.")
        print("Use a compiler that matches Python architecture and rebuild.")
        return 1

    print(f"Built and loaded {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
