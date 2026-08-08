from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

BINDING_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = BINDING_DIRECTORY.parents[2]


class BuildWithLicense(build_py):
    def run(self) -> None:
        super().run()
        shutil.copy2(
            REPOSITORY_ROOT / "LICENSE",
            Path(self.build_lib) / "vinary_tree_interop" / "LICENSE",
        )


setup(cmdclass={"build_py": BuildWithLicense})
