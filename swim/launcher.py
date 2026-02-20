# swim/launcher.py

"""Launch all SWIM A2A agent servers as subprocesses."""

import logging
import signal
import subprocess
import sys
import time
from typing import List

from swim.shared.a2a_config import AGENT_PORTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("SWIM-Launcher")

AGENT_MODULES = {
    "homogen": "swim.agents.homogen.a2a_server",
    "calibro": "swim.agents.calibro.a2a_server",
    "visios": "swim.agents.visios.a2a_server",
    "predikt": "swim.agents.predikt.a2a_server",
    "orchestrator": "swim.agents.orchestrator.a2a_server",
}


def main():
    processes: List[subprocess.Popen] = []

    def shutdown(signum, frame):
        logger.info("Shutting down all agents...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait(timeout=10)
        logger.info("All agents stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start agent servers with staggered delay
    for name, module in AGENT_MODULES.items():
        port = AGENT_PORTS[name]
        logger.info("Starting %s on port %d ...", name.upper(), port)
        proc = subprocess.Popen(
            [sys.executable, "-m", module],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        processes.append(proc)
        time.sleep(1)

    logger.info(
        "All SWIM agents running. Ports: %s",
        ", ".join(f"{n}={p}" for n, p in AGENT_PORTS.items()),
    )

    # Wait for any process to exit
    try:
        while True:
            for i, proc in enumerate(processes):
                retcode = proc.poll()
                if retcode is not None:
                    agent_name = list(AGENT_MODULES.keys())[i]
                    logger.warning("%s exited with code %d", agent_name.upper(), retcode)
            time.sleep(2)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
