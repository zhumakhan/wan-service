import signal
import logging


class GracefulKiller:
    shutdown = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logging.warning("Received shutdown event")
        self.shutdown = True
