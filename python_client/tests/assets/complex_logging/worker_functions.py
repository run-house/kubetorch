"""Worker class and functions that use the configured logging."""
import logging
import time

from .logging_config import logger  # Use the configured logger


class LoggingTestWorker:
    """
    Worker class that tests various logging scenarios with complex structlog configuration.
    """

    def __init__(self):
        self.class_logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.class_logger.info("LoggingTestWorker initialized")

    def process_with_logs(self, num_iterations=3):
        """
        Method that performs processing with various types of logging output.
        Tests that both print statements and logger calls work correctly with
        complex structlog configuration.
        """
        # Also get a local logger to test __name__ based loggers
        local_logger = logging.getLogger(__name__)

        results = []
        for i in range(num_iterations):
            # Test regular print statements
            print(f"Processing iteration {i}")

            # Test module-level logger from logging_config
            logger.info(f"Module logger info: Processing item {i}")

            # Test local logger
            local_logger.info(f"Local logger info: Item {i} in progress")
            local_logger.warning(f"Local logger warning: Check item {i}")

            # Test class logger
            self.class_logger.info(f"Class logger: Processing {i}")

            # Simulate some work
            time.sleep(0.5)
            results.append(i * 2)

            # Test debug level (should not appear unless log level is DEBUG)
            local_logger.debug(f"Debug details for item {i}")

        # Final summary logs
        print(f"Completed {num_iterations} iterations")
        logger.info(f"Final results: {results}")

        return results

    def process_with_errors(self, should_fail=False):
        """
        Method that tests error logging and exception handling.
        """
        error_logger = logging.getLogger(f"{__name__}.errors")

        try:
            error_logger.info("Starting process with potential errors")

            if should_fail:
                error_logger.error("About to raise an exception")
                raise ValueError("Intentional test error")

            error_logger.info("Process completed successfully")
            return "success"

        except Exception as e:
            error_logger.exception(f"Process failed with error: {e}")
            raise

    def nested_logging_test(self):
        """
        Test that nested function calls with different loggers work correctly.
        """
        parent_logger = logging.getLogger("parent.logger")

        def inner_function():
            child_logger = logging.getLogger("parent.logger.child")
            child_logger.info("Log from inner function")
            print("Print from inner function")
            return "inner_result"

        parent_logger.info("Starting nested test")
        result = inner_function()
        parent_logger.info(f"Nested test complete with result: {result}")
        self.class_logger.info(f"Class logger confirms: {result}")

        return result
