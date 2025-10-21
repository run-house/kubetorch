"""Module that sets up complex structlog configuration."""
import logging
import logging.config

try:
    import structlog

    # Define structlog processors
    processors = [
        # log level / logger name, effects coloring in ConsoleRenderer(colors=True)
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        # timestamp format
        structlog.processors.TimeStamper(fmt="iso"),
        # To see all CallsiteParameterAdder options:
        # https://www.structlog.org/en/stable/api.html#structlog.processors.CallsiteParameterAdder
        # more options include module, pathname, process, process_name, thread, thread_name
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        # Any structlog.contextvars.bind_contextvars included in middleware/functions
        structlog.contextvars.merge_contextvars,
        # strip _record and _from_structlog keys from event dictionary
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    ]

    # Configure standard logging with structlog formatter
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": processors,
                    "foreign_pre_chain": [structlog.stdlib.ExtraAdder()],
                },
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                # Root / default logger set to INFO (applies to all third party loggers)
                "": {
                    "handlers": ["default"],
                    "level": "INFO",
                    "propagate": True,
                },
            },
        }
    )

    # Configure structlog itself
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),  # <- This is the magic
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
except ImportError:
    pass

# Create module-level logger
logger = logging.getLogger(__name__)
