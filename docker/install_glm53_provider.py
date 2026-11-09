"""Install pinned GLM model packages without replacing the release Bridge core."""

import subprocess
import sys
from pathlib import Path

DEPENDENCY_ROOT = Path("/opt/miles-glm53-bridge")
MODELS_ROOT = Path("/opt/miles-glm53-models")


def main():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--upgrade",
            "--target",
            str(DEPENDENCY_ROOT),
            "-r",
            str(Path(__file__).with_name("requirements-glm53-provider.txt")),
        ],
        check=True,
    )
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    # Only these model packages enter Bridge's search path. PEFT, conversion,
    # other providers and distribution metadata stay on the release revision.
    for name in ("glm5", "glm5_next"):
        source = DEPENDENCY_ROOT / "megatron" / "bridge" / "models" / name
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination = MODELS_ROOT / name
        if destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source, target_is_directory=True)


if __name__ == "__main__":
    main()
