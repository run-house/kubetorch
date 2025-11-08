from dataclasses import asdict, dataclass

from kubetorch.logger import get_logger

logger = get_logger(__name__)


class AutoScalingError(Exception):
    pass


@dataclass
class AutoscalingConfig:
    # The concurrent requests or requests per second threshold that triggers scaling
    target: int = None

    # The time window used to calculate the average of metrics for scaling decisions
    window: str = None

    # The metric type to base scaling decisions on:
    # - concurrency: number of simultaneous requests
    # - rps: requests per second
    # - cpu: CPU utilization (requires HPA class)
    # - memory: Memory utilization (requires HPA class)
    metric: str = None

    # The percentage of the target value at which to start scaling.
    # E.g., if target=100 and target_utilization=70, scaling occurs at 70 requests
    target_utilization: int = None

    # Minimum number of replicas. 0 allows scaling to zero when idle
    min_scale: int = None

    # Maximum number of replicas the service can scale up to
    max_scale: int = None

    # Initial number of pods launched by the service
    initial_scale: int = None

    # Maximum concurrent requests per pod (containerConcurrency).
    # If not set, pods accept unlimited concurrent requests.
    concurrency: int = None

    # Time to keep the last pod before scaling to zero (e.g., "30s", "1m5s")
    scale_to_zero_pod_retention_period: str = None

    # Delay before scaling down (e.g., "15m"). Only for KPA autoscaler.
    scale_down_delay: str = None

    # Autoscaler class: "kpa.autoscaling.knative.dev" or "hpa.autoscaling.knative.dev"
    autoscaler_class: str = None

    # Progress deadline for deployment (e.g., "10m"). Time to wait for deployment to be ready.
    progress_deadline: str = None

    def __init__(self, **kwargs):
        """Support additional kwargs for autoscaling annotations"""
        for field in self.__annotations__:
            setattr(self, field, kwargs.pop(field, getattr(self, field, None)))

        # set additional kwargs as annotations
        self.extra_annotations = {f"autoscaling.knative.dev/{k}": str(v) for k, v in kwargs.items()}

        self._validate()

    def _validate(self):
        """Validation logic moved to separate method"""
        if self.min_scale is not None and self.max_scale is not None and self.min_scale > self.max_scale:
            raise AutoScalingError("min_scale cannot be greater than max_scale")
        if self.window is not None and not self.window.endswith(("s", "m", "h")):
            raise AutoScalingError("window must end with s, m, or h")
        if self.target_utilization is not None and (self.target_utilization <= 0 or self.target_utilization > 100):
            raise AutoScalingError("target_utilization must be between 1 and 100")
        if self.scale_to_zero_pod_retention_period is not None:
            # Validate time format (e.g., "30s", "1m5s", "2h")
            import re

            if not re.match(r"^\d+[smh](\d+[smh])*$", self.scale_to_zero_pod_retention_period):
                raise AutoScalingError(
                    "scale_to_zero_pod_retention_period must be a valid duration (e.g., '30s', '1m5s')"
                )
        if self.scale_down_delay is not None:
            # Validate time format
            import re

            if not re.match(r"^\d+[smh](\d+[smh])*$", self.scale_down_delay):
                raise AutoScalingError("scale_down_delay must be a valid duration (e.g., '15m', '1h')")
        if self.autoscaler_class is not None and self.autoscaler_class not in [
            "kpa.autoscaling.knative.dev",
            "hpa.autoscaling.knative.dev",
        ]:
            raise AutoScalingError(
                "autoscaler_class must be 'kpa.autoscaling.knative.dev' or 'hpa.autoscaling.knative.dev'"
            )
        if self.progress_deadline is not None:
            # Validate time format
            import re

            if not re.match(r"^\d+[smh](\d+[smh])*$", self.progress_deadline):
                raise AutoScalingError("progress_deadline must be a valid duration (e.g., '10m', '600s')")

    def __post_init__(self):
        """Call the same validation for dataclass initialization"""
        self._validate()

    def dict(self):
        return asdict(self)

    def convert_to_annotations(self) -> dict:
        """Convert config to a dictionary of annotations for Knative"""
        annotations = {}

        # Set autoscaler class if specified, otherwise use default KPA
        if self.autoscaler_class is not None:
            annotations["autoscaling.knative.dev/class"] = self.autoscaler_class
        else:
            annotations["autoscaling.knative.dev/class"] = "kpa.autoscaling.knative.dev"

        # Only set annotations for values that were explicitly provided
        if self.target is not None:
            annotations["autoscaling.knative.dev/target"] = str(self.target)

        if self.min_scale is not None:
            annotations["autoscaling.knative.dev/min-scale"] = str(self.min_scale)

        if self.max_scale is not None:
            annotations["autoscaling.knative.dev/max-scale"] = str(self.max_scale)

        if self.window is not None:
            annotations["autoscaling.knative.dev/window"] = self.window

        if self.metric is not None:
            annotations["autoscaling.knative.dev/metric"] = self.metric

        if self.target_utilization is not None:
            annotations["autoscaling.knative.dev/target-utilization-percentage"] = str(self.target_utilization)

        if self.initial_scale is not None:
            annotations["autoscaling.knative.dev/initial-scale"] = str(self.initial_scale)

        if self.scale_to_zero_pod_retention_period is not None:
            annotations[
                "autoscaling.knative.dev/scale-to-zero-pod-retention-period"
            ] = self.scale_to_zero_pod_retention_period

        if self.scale_down_delay is not None:
            annotations["autoscaling.knative.dev/scale-down-delay"] = self.scale_down_delay

        # Add any extra annotations from the config
        if hasattr(self, "extra_annotations"):
            annotations.update(self.extra_annotations)

        return annotations
