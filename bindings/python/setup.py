from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py

BINDING_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = BINDING_DIRECTORY.parents[1]


def native_library_name() -> str:
    system = platform.system()
    if system == "Windows":
        return "liblevenshtein.dll"
    if system == "Darwin":
        return "libliblevenshtein.dylib"
    return "libliblevenshtein.so"


def native_library() -> Path:
    explicit = os.environ.get("LIBLEVENSHTEIN_PREBUILT_LIBRARY")
    if explicit:
        library = Path(explicit).expanduser().resolve()
        if not library.is_file():
            raise FileNotFoundError(
                f"LIBLEVENSHTEIN_PREBUILT_LIBRARY is not a file: {library}"
            )
        return library

    command = [
        "cargo",
        "build",
        "--manifest-path",
        str(REPOSITORY_ROOT / "Cargo.toml"),
        "--release",
        "--features",
        "python-bindings",
    ]
    target = os.environ.get("LIBLEVENSHTEIN_RUST_TARGET")
    if target:
        command.extend(["--target", target])
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)

    target_directory = Path(
        os.environ.get("CARGO_TARGET_DIR", REPOSITORY_ROOT / "target")
    )
    profile_directory = (
        target_directory / target / "release"
        if target
        else target_directory / "release"
    )
    library = profile_directory / native_library_name()
    if not library.is_file():
        raise FileNotFoundError(f"Cargo did not produce the native library: {library}")
    return library


class BuildWithNativeLibrary(build_py):
    def run(self) -> None:
        super().run()
        destination = Path(self.build_lib) / "liblevenshtein" / "native"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(native_library(), destination / native_library_name())
        shutil.copy2(
            REPOSITORY_ROOT / "LICENSE",
            Path(self.build_lib) / "liblevenshtein" / "LICENSE",
        )


class PlatformDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


class PortablePythonPlatformWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


setup(
    cmdclass={
        "build_py": BuildWithNativeLibrary,
        "bdist_wheel": PortablePythonPlatformWheel,
    },
    distclass=PlatformDistribution,
)
