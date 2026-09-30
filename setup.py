"""Packaging for the GTA FBX -> IFP retarget engine.

Windows, from a checkout::

    py -3 -m venv .venv
    .venv\\Scripts\\pip install -e .
    .venv\\Scripts\\gtafbx doctor

Without installing, straight from the source tree::

    py -3 cli.py inspect-fbx dance.fbx

Standalone .exe (no Python needed on the target machine)::

    py -3 -m pip install pyinstaller
    build_exe.bat
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent


def read(name: str) -> str:
    path = ROOT / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def version() -> str:
    for line in (ROOT / "gta_fbx_ifp_converter" / "__init__.py").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "0.0.0"


if sys.version_info < (3, 10):
    raise SystemExit(
        "This tool needs Python 3.10 or newer "
        f"(running {sys.version.split()[0]})."
    )

setup(
    name="gta-fbx-ifp-converter",
    version=version(),
    description=(
        "Retarget an FBX character animation onto a GTA San Andreas / MTA:SA "
        "ped skeleton and write an ANP3 IFP"
    ),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["gta_fbx_ifp_converter", "gta_fbx_ifp_converter.*"]),
    py_modules=["cli"],
    install_requires=[
        "numpy>=1.24",
        # MIT; DFF/HAnim/IFP handling is delegated to it rather than
        # re-implemented, so the GTA core stays small and testable.
        "rwfury>=0.6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
        "build": ["pyinstaller>=6.0"],
    },
    entry_points={
        "console_scripts": [
            "gtafbx = cli:main",
            "gta-fbx-ifp = cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
    ],
    keywords="gta san andreas mta ifp anp3 dff fbx retarget animation",
    zip_safe=False,
)
