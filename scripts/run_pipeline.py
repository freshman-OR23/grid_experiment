from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    python_executable = "E:\\Anaconda\\envs\\torch_gpu\\python.exe"

    commands = [
        [python_executable, str(project_root / "scripts" / "run_tokenizer.py"), "--config", args.config],
        [python_executable, str(project_root / "scripts" / "run_train.py"), "--config", args.config],
        [python_executable, str(project_root / "scripts" / "run_eval.py"), "--config", args.config, "--split", "test"],
    ]
    if args.smoke_test:
        commands = [command + ["--smoke-test"] for command in commands]

    for command in commands:
        subprocess.run(command, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
