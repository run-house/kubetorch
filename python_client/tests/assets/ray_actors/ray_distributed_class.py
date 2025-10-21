import time
from typing import List


class RayDistributedProcessor:
    """Test class for Ray distributed processing with tasks and actors."""

    def __init__(self, num_workers=2):
        import ray

        # Initialize Ray (should connect to existing cluster)
        ray.init(address="auto")
        self.num_workers = num_workers
        self.worker_pool = []

        # Create a pool of worker actors
        for i in range(num_workers):
            worker = WorkerActor.remote(worker_id=i)
            self.worker_pool.append(worker)

    def process_with_tasks(self, data: List[int]) -> dict:
        """Test Ray tasks for stateless parallel processing."""
        import ray

        # Process data using Ray tasks (stateless)
        futures = []
        for item in data:
            future = process_item_task.remote(item)
            futures.append(future)

        # Wait for all tasks to complete
        results = ray.get(futures)

        return {
            "task_results": results,
            "total_processed": len(results),
            "sum": sum(results),
        }

    def process_with_actors(self, data: List[int]) -> dict:
        """Test Ray actors for stateful parallel processing."""
        import ray

        # Distribute work across worker actors (stateful)
        futures = []
        for i, item in enumerate(data):
            worker = self.worker_pool[i % len(self.worker_pool)]
            future = worker.process_item.remote(item)
            futures.append(future)

        # Wait for all actor calls to complete
        results = ray.get(futures)

        # Get state from all workers
        state_futures = [worker.get_state.remote() for worker in self.worker_pool]
        worker_states = ray.get(state_futures)

        return {
            "actor_results": results,
            "worker_states": worker_states,
            "total_processed": len(results),
            "sum": sum(results),
        }

    def mixed_processing(self, data: List[int]) -> dict:
        """Test combining Ray tasks and actors."""
        import ray

        # Step 1: Use tasks to preprocess data
        preprocess_futures = [preprocess_task.remote(item) for item in data]
        preprocessed_data = ray.get(preprocess_futures)

        # Step 2: Use actors to process the preprocessed data
        actor_futures = []
        for i, item in enumerate(preprocessed_data):
            worker = self.worker_pool[i % len(self.worker_pool)]
            future = worker.process_item.remote(item)
            actor_futures.append(future)

        actor_results = ray.get(actor_futures)

        # Step 3: Use tasks to postprocess results
        postprocess_futures = [
            postprocess_task.remote(result) for result in actor_results
        ]
        final_results = ray.get(postprocess_futures)

        return {
            "mixed_results": final_results,
            "preprocessing_done": len(preprocessed_data),
            "actor_processing_done": len(actor_results),
            "postprocessing_done": len(final_results),
            "final_sum": sum(final_results),
        }

    def get_cluster_info(self) -> dict:
        """Get information about the Ray cluster."""
        import ray

        return {
            "cluster_resources": ray.cluster_resources(),
            "available_resources": ray.available_resources(),
            "num_workers": len(self.worker_pool),
            "ray_nodes": len(ray.nodes()),
        }

    def cleanup(self):
        """Clean up Ray resources."""
        import ray

        # Kill all worker actors
        for worker in self.worker_pool:
            ray.kill(worker)

        self.worker_pool = []


# Ray remote actor for stateful processing
import ray


@ray.remote
class WorkerActor:
    """Ray actor for stateful processing."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.processed_count = 0
        self.total_value = 0

    def process_item(self, item: int) -> int:
        """Process an item and update internal state."""
        time.sleep(0.05)  # Simulate some work
        result = item * 2 + self.worker_id

        # Update state
        self.processed_count += 1
        self.total_value += result

        return result

    def get_state(self) -> dict:
        """Get the current state of this worker."""
        return {
            "worker_id": self.worker_id,
            "processed_count": self.processed_count,
            "total_value": self.total_value,
        }


# Ray remote tasks for stateless processing
@ray.remote
def process_item_task(item: int) -> int:
    """Stateless task to process an item."""
    time.sleep(0.05)  # Simulate some work
    return item * 3


@ray.remote
def preprocess_task(item: int) -> int:
    """Preprocess an item."""
    return item + 10


@ray.remote
def postprocess_task(item: int) -> int:
    """Postprocess an item."""
    return item - 5
