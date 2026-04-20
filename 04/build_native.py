from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE = SCRIPT_DIR / "native_intersect.c"
OUTPUT = SCRIPT_DIR / ("native_intersect.dll" if sys.platform == "win32" else "native_intersect.so")


def main() -> int:
    compiler = shutil.which("gcc")
    if compiler is None:
        print("gcc was not found in PATH.")
        print("Install a 64-bit compiler, for example MSVC Build Tools or MinGW-w64 x86_64.")
        return 1

    command = [compiler, "-O3", "-shared"]
    if sys.platform != "win32":
        command.append("-fPIC")
    command.extend(["-o", str(OUTPUT), str(SOURCE)])

    print("Python:", platform.architecture()[0])
    print("Compiler:", compiler)
    print("Command:", " ".join(command))
    subprocess.run(command, check=True)

    try:
        ctypes.CDLL(str(OUTPUT))
    except OSError as error:
        print(f"Built {OUTPUT}, but Python could not load it:")
        print(error)
        OUTPUT.unlink(missing_ok=True)
        print(f"Removed incompatible {OUTPUT}")
        print("Most likely cause on Windows: 32-bit gcc with 64-bit Python.")
        print("Use a 64-bit compiler and rebuild.")
        return 1

    print(f"Built and loaded {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
