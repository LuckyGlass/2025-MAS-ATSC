"""
Define logger.
@author: Yansheng Mao
"""
import contextlib
import logging
import os
import sys

_LOGGER = None

def get_logger():
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    
    name = "atsc"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    _LOGGER = logger
    return logger


@contextlib.contextmanager
def suppress_all_output():
    # Save original file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Open null device
    with open(os.devnull, 'w') as devnull:
        # Flush Python buffers
        sys.stdout.flush()
        sys.stderr.flush()

        # Save original dup FDs so we can restore
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        try:
            # Redirect stdout and stderr to /dev/null at OS level
            os.dup2(devnull.fileno(), original_stdout_fd)
            os.dup2(devnull.fileno(), original_stderr_fd)
            yield
        finally:
            # Restore original file descriptors
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)

            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)