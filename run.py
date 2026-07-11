from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


SCRIPT_MAP = {
    "windows": {
        "full": {
            "prepare": PROJECT_ROOT / "prepare_windows_full.bat",
            "train": PROJECT_ROOT / "train_full_windows.bat",
            "run": PROJECT_ROOT / "run_windows_full.bat",
        },
    },
    "linux": {
        "full": {
            "prepare": PROJECT_ROOT / "prepare_linux_full.sh",
            "train": PROJECT_ROOT / "train_full_linux.sh",
            "run": PROJECT_ROOT / "run_linux_full.sh",
        },
    },
}


TARGETS = ["auto", "windows", "linux"]
MODES = ["full"]
ACTIONS = ["prepare", "train", "run"]


def windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()

    if not drive:
        raise RuntimeError(f"Cannot convert path to WSL path: {resolved}")

    relative = resolved.relative_to(resolved.anchor)
    parts = "/".join(relative.parts)

    return f"/mnt/{drive}/{parts}"


def detect_target() -> str:
    if platform.system().lower() == "windows":
        return "windows"

    return "linux"


def parse_env_override(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Environment overrides must use KEY=VALUE syntax."
        )

    key, env_value = value.split("=", 1)
    key = key.strip()

    if not key:
        raise argparse.ArgumentTypeError(
            "Environment override keys cannot be empty."
        )

    return key, env_value


def resolve_mode_action(
    mode_or_action: str,
    action: str | None,
) -> tuple[str, str]:
    if action is None:
        mode = "full"
        resolved_action = mode_or_action
    else:
        mode = mode_or_action
        resolved_action = action

    if mode not in MODES:
        raise argparse.ArgumentTypeError(
            f"mode must be one of {', '.join(MODES)}"
        )

    if resolved_action not in ACTIONS:
        raise argparse.ArgumentTypeError(
            f"action must be one of {', '.join(ACTIONS)}"
        )

    return mode, resolved_action


def build_command(
    target: str,
    script: Path,
    explicit_env: dict[str, str],
) -> list[str]:
    if target == "windows":
        cmd = shutil.which("cmd.exe")

        if cmd is None:
            raise RuntimeError(
                "cmd.exe was not found. Run this target from Windows, "
                "or choose the linux target."
            )

        return [cmd, "/c", str(script)]

    if os.name == "nt":
        wsl = shutil.which("wsl.exe")

        if wsl is not None:
            wsl_env = explicit_env.copy()
            wsl_env["PROJECT_ROOT"] = windows_to_wsl_path(PROJECT_ROOT)
            env_args = [f"{key}={value}" for key, value in wsl_env.items()]

            return [
                wsl,
                "env",
                *env_args,
                "bash",
                windows_to_wsl_path(script),
            ]

    bash = shutil.which("bash")

    if bash is None:
        raise RuntimeError(
            "bash was not found. Install bash or choose the windows target."
        )

    return [bash, str(script)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run diffusion-bd platform workflows from one portable "
            "command-line entry point."
        )
    )
    parser.add_argument(
        "target",
        choices=TARGETS,
        help="Platform workflow to run. auto uses the current OS.",
    )
    parser.add_argument(
        "mode_or_action",
        help=(
            "Experiment mode (full) or, for short commands, "
            "the action to run."
        ),
    )
    parser.add_argument(
        "action",
        nargs="?",
        help="Action to run when mode is provided: prepare, train, or run.",
    )
    parser.add_argument(
        "--set",
        dest="env_overrides",
        action="append",
        default=[],
        type=parse_env_override,
        metavar="KEY=VALUE",
        help=(
            "Set or override an environment variable for the child "
            "process. Can be repeated."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would run without executing it.",
    )

    args = parser.parse_args()

    target = detect_target() if args.target == "auto" else args.target

    try:
        mode, action = resolve_mode_action(
            args.mode_or_action,
            args.action,
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    script = SCRIPT_MAP[target][mode][action]

    if not script.exists():
        print(f"ERROR: script not found: {script}", file=sys.stderr)
        return 1

    explicit_env = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "BD_TARGET": target,
        "BD_MODE": mode,
        "BD_ACTION": action,
    }

    for key, value in args.env_overrides:
        explicit_env[key] = value

    try:
        command = build_command(target, script, explicit_env)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env.update(explicit_env)

    printable_command = " ".join(
        f'"{part}"' if " " in part else part
        for part in command
    )

    print("Project root:", PROJECT_ROOT)
    print("Target:", target)
    print("Mode:", mode)
    print("Action:", action)
    print("Command:", printable_command)
    sys.stdout.flush()

    if args.dry_run:
        return 0

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
