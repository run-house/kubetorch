import time


def dummy_objective_function(width, height, step):
    """Dummy objective function for hyperparameter optimization."""
    # Simulate some computation time
    time.sleep(0.1)
    # Return a score based on the parameters
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


def ray_tune_hpo(num_samples=4, max_concurrent_trials=2):
    """Ray Tune hyperparameter optimization function for testing."""
    import ray
    from ray import tune

    # Initialize Ray (should connect to existing cluster)
    ray.init(address="auto")

    def train_function(config):
        """Training function for Ray Tune."""
        step = 0
        for step in range(3):  # Short training for testing
            score = dummy_objective_function(config["width"], config["height"], step)
            # Report the score to Tune
            tune.report(dict(score=score, step=step))

    # Define the search space
    search_space = {
        "width": tune.uniform(0, 10),
        "height": tune.uniform(-10, 10),
    }

    # Create and run the tuner
    tuner = tune.Tuner(
        train_function,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="max",
            max_concurrent_trials=max_concurrent_trials,
            num_samples=num_samples,
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best_result = results.get_best_result()

    # Return summary of the HPO run
    return {
        "best_score": best_result.metrics["score"],
        "best_config": best_result.config,
        "num_trials": len(results),
        "status": "completed",
    }
