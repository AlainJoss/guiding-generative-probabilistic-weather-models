import subprocess
from itertools import product

from src.constants import GUIDANCE_MODES


CONFIG_TYPE = "guided"
TEST_FLAG = False
ALPHAS = [2.0]
WS = [5.0, 10.0, 15.0]


def run_command(guidance_mode: str, alpha: float, w: float) -> None:
    cmd = [
        "python",
        "-m",
        "src.runners.run_all_configs",
        "--config-type",
        CONFIG_TYPE,
        "--guidance-mode",
        guidance_mode,
        "--alpha",
        str(alpha),
        "--w",
        str(w),
    ]

    if TEST_FLAG:
        cmd.append("--test")

    print(f"Running: mode={guidance_mode}, alpha={alpha}, w={w}")
    subprocess.run(cmd, check=True)


def main() -> None:
    for guidance_mode, alpha, w in product(GUIDANCE_MODES, ALPHAS, WS):
        run_command(guidance_mode, alpha, w)

    print("Done. Guidance-mode and lambda-schedule ablation finished.")


if __name__ == "__main__":
    main()