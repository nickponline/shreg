import logging


def get_custom_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # Custom formatter
    formatter = logging.Formatter("%(message)s")
    # formatter = logging.Formatter(
    #     "%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )
    handler.setFormatter(formatter)

    # Add handler to the logger
    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger


# Usage example
if __name__ == "__main__":
    log = get_custom_logger(__name__)
    log.debug("This is a debug message")
    log.info("This is an info message")
    log.warning("This is a warning message")
    log.error("This is an error message")
    log.critical("This is a critical message")
