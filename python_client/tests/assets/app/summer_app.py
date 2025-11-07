import argparse
import logging

import kubetorch as kt


image = kt.Image("docker-latest")
kwargs = {"cpus": "0.01"}
app = kt.app(image=image, **kwargs)


def get_test_logger(name=None):
    """Use a generic logger for testing that doesnt require a kubetorch dependency."""
    logger = logging.getLogger(name or __name__)

    # Avoid adding handlers if they already exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


logger = get_test_logger()


def summer(a, b):
    print(f"Hello from the cluster stdout! {a} {b}")
    logger.info(f"Hello from the cluster logs! {a} {b}")

    return a + b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested", action="store_true", help="Whether to run nested instead of base.")
    parser.add_argument("arg1", type=int)
    parser.add_argument("arg2", type=int)
    args = parser.parse_args()

    if args.nested:
        compute = kt.Compute(image=image, **kwargs)
        remote_summer = kt.fn(summer).to(compute)
        result = remote_summer(args.arg1, args.arg2, stream_logs=True)

        print(f"result: {result}")
    else:
        summer(args.arg1, args.arg2)
