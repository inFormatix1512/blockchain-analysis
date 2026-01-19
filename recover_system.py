"""Automatic recovery script for database + ingest orchestration."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecoverySettings:
    postgres_ready_cmd: str = "docker compose exec -T postgres pg_isready -U postgres"
    bitcoind_health_cmd: str = "docker inspect --format='{{.State.Health.Status}}' bitcoind"
    ingest_start_cmd: str = "docker compose up -d ingest"
    orchestrator_script: str = "ingest/ops/run_sampling_campaign.py"
    check_interval_seconds: int = 10
    bitcoind_timeout_seconds: int = 20 * 60
    cool_down_seconds: int = 5


def run_cmd(cmd: str) -> Tuple[int, str, str]:
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except Exception as exc:
        return -1, "", str(exc)


def check_postgres(settings: RecoverySettings) -> bool:
    code, out, err = run_cmd(settings.postgres_ready_cmd)
    if code == 0:
        return True
    logger.info("Postgres wait: %s %s", out, err)
    return False


def check_bitcoind(settings: RecoverySettings) -> bool:
    code, out, _ = run_cmd(settings.bitcoind_health_cmd)
    if code == 0 and "healthy" in out:
        return True
    logger.info("Bitcoind status: %s", out)
    return False


def start_ingest(settings: RecoverySettings) -> None:
    logger.info("Starting ingest container...")
    run_cmd(settings.ingest_start_cmd)


def start_orchestrator(settings: RecoverySettings) -> None:
    logger.info("Checking if orchestrator is already running...")
    code, _, _ = run_cmd("pgrep -f run_sampling_campaign.py")
    if code == 0:
        logger.info("Orchestrator already running.")
        return

    logger.info("Starting orchestrator (%s)...", settings.orchestrator_script)
    with open("sampling.log", "a") as outfile:
        subprocess.Popen(
            ["nohup", "python3", "-u", settings.orchestrator_script],
            stdout=outfile,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp,
        )


def main(settings: RecoverySettings | None = None) -> None:
    settings = settings or RecoverySettings()
    logger.info("--- AUTOMATIC RECOVERY SYSTEM STARTED ---")

    logger.info("Step 1: Waiting for Postgres...")
    while not check_postgres(settings):
        time.sleep(settings.check_interval_seconds)
    logger.info("Postgres is READY!")

    logger.info("Step 2: Waiting for Bitcoind to be healthy...")
    timeout_loops = settings.bitcoind_timeout_seconds // settings.check_interval_seconds
    for _ in range(timeout_loops):
        if check_bitcoind(settings):
            logger.info("Bitcoind is HEALTHY!")
            break
        time.sleep(settings.check_interval_seconds)
    else:
        logger.warning("Bitcoind not healthy after timeout. Trying to proceed anyway...")

    time.sleep(settings.cool_down_seconds)

    logger.info("Step 3: Starting Ingest Docker Service...")
    start_ingest(settings)

    time.sleep(settings.cool_down_seconds)

    logger.info("Step 4: Starting Orchestrator Script...")
    start_orchestrator(settings)

    logger.info("--- RECOVERY PROCEDURE COMPLETED ---")


if __name__ == "__main__":
    main()
