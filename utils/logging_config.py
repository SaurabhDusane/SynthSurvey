"""Central logging configuration.

Modules create loggers with ``logging.getLogger(__name__)`` but nothing emits
them until a handler is configured. ``setup_logging`` wires that up once at app
startup; it is idempotent so Streamlit reruns don't stack duplicate handlers.
"""

import logging

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger a single time.

    Args:
        level: A logging level name (e.g. "DEBUG", "INFO", "WARNING").
    """
    global _CONFIGURED
    resolved = getattr(logging, str(level).upper(), logging.INFO)

    root = logging.getLogger()
    if not _CONFIGURED and not root.handlers:
        logging.basicConfig(
            level=resolved,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _CONFIGURED = True
    else:
        root.setLevel(resolved)
