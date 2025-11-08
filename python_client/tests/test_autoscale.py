import pytest

from kubetorch.serving.autoscaling import AutoscalingConfig, AutoScalingError


@pytest.mark.level("unit")
def test_default_autoscale():
    """Test that default config only sets the autoscaler class"""
    config = AutoscalingConfig()
    annotations = config.convert_to_annotations()
    # Should only have the autoscaler class, no other defaults
    assert annotations == {"autoscaling.knative.dev/class": "kpa.autoscaling.knative.dev"}


@pytest.mark.level("unit")
def test_autoscale_with_min_scale():
    """Test that min_scale is passed through correctly"""
    config = AutoscalingConfig(min_scale=0)
    annotations = config.convert_to_annotations()
    assert annotations["autoscaling.knative.dev/min-scale"] == "0"

    config = AutoscalingConfig(min_scale=3)
    annotations = config.convert_to_annotations()
    assert annotations["autoscaling.knative.dev/min-scale"] == "3"


@pytest.mark.level("unit")
def test_autoscale_with_scale_to_zero_pod_retention():
    """Test the new scale_to_zero_pod_retention_period parameter"""
    config = AutoscalingConfig(scale_to_zero_pod_retention_period="30s")
    annotations = config.convert_to_annotations()
    assert annotations["autoscaling.knative.dev/scale-to-zero-pod-retention-period"] == "30s"

    config = AutoscalingConfig(scale_to_zero_pod_retention_period="1m5s")
    annotations = config.convert_to_annotations()
    assert annotations["autoscaling.knative.dev/scale-to-zero-pod-retention-period"] == "1m5s"


@pytest.mark.level("unit")
def test_autoscale_with_all_parameters():
    """Test that all parameters are passed through correctly"""
    config = AutoscalingConfig(
        target=50,
        window="120s",
        metric="rps",
        target_utilization=80,
        min_scale=2,
        max_scale=10,
        initial_scale=3,
        concurrency=100,
        scale_to_zero_pod_retention_period="45s",
        scale_down_delay="10m",
        autoscaler_class="hpa.autoscaling.knative.dev",
    )
    annotations = config.convert_to_annotations()

    assert annotations["autoscaling.knative.dev/class"] == "hpa.autoscaling.knative.dev"
    assert annotations["autoscaling.knative.dev/target"] == "50"
    assert annotations["autoscaling.knative.dev/window"] == "120s"
    assert annotations["autoscaling.knative.dev/metric"] == "rps"
    assert annotations["autoscaling.knative.dev/target-utilization-percentage"] == "80"
    assert annotations["autoscaling.knative.dev/min-scale"] == "2"
    assert annotations["autoscaling.knative.dev/max-scale"] == "10"
    assert annotations["autoscaling.knative.dev/initial-scale"] == "3"
    assert annotations["autoscaling.knative.dev/scale-to-zero-pod-retention-period"] == "45s"
    assert annotations["autoscaling.knative.dev/scale-down-delay"] == "10m"
    # Note: concurrency doesn't appear in annotations, it's handled separately for containerConcurrency


@pytest.mark.level("unit")
def test_autoscale_validation_errors():
    """Test that validation catches invalid inputs"""
    with pytest.raises(AutoScalingError, match="min_scale cannot be greater than max_scale"):
        AutoscalingConfig(min_scale=5, max_scale=2)

    with pytest.raises(AutoScalingError, match="window must end with s, m, or h"):
        AutoscalingConfig(window="60")

    with pytest.raises(AutoScalingError, match="target_utilization must be between 1 and 100"):
        AutoscalingConfig(target_utilization=0)

    with pytest.raises(AutoScalingError, match="target_utilization must be between 1 and 100"):
        AutoscalingConfig(target_utilization=101)

    with pytest.raises(
        AutoScalingError,
        match="scale_to_zero_pod_retention_period must be a valid duration",
    ):
        AutoscalingConfig(scale_to_zero_pod_retention_period="invalid")

    with pytest.raises(AutoScalingError, match="scale_down_delay must be a valid duration"):
        AutoscalingConfig(scale_down_delay="10")

    with pytest.raises(AutoScalingError, match="autoscaler_class must be"):
        AutoscalingConfig(autoscaler_class="invalid.class")


@pytest.mark.level("unit")
def test_autoscale_timing_defaults():
    """Test that autoscaled services get appropriate timing defaults"""
    import kubetorch as kt

    # All workloads should get timing defaults (CPU or GPU)
    cpu_compute = kt.Compute(cpus="2", memory="4Gi")
    cpu_compute.autoscale(target=10)

    # Check that timing defaults were applied
    assert cpu_compute.autoscaling_config.scale_down_delay == "1m"
    assert cpu_compute.autoscaling_config.scale_to_zero_pod_retention_period == "10m"
    # Default launch_timeout is 900s, so progress_deadline should be 1080s (20% buffer)
    assert cpu_compute.autoscaling_config.progress_deadline == "1080s"
    # Other parameters should not be defaulted
    assert cpu_compute.autoscaling_config.concurrency is None
    assert cpu_compute.autoscaling_config.min_scale is None

    # Test with small launch_timeout - should use default 10m
    compute_small_timeout = kt.Compute(cpus="1", launch_timeout=300)  # 5 minutes
    compute_small_timeout.autoscale(target=5)
    # 300 * 1.2 = 360s, which is less than 600s (10m), so use default
    assert compute_small_timeout.autoscaling_config.progress_deadline == "10m"

    # Test with large launch_timeout - progress_deadline should be adjusted
    compute_large_timeout = kt.Compute(cpus="1", launch_timeout=1200)  # 20 minutes
    compute_large_timeout.autoscale(target=5)
    # 1200 * 1.2 = 1440 seconds
    assert compute_large_timeout.autoscaling_config.progress_deadline == "1440s"

    # Verify defaults can be overridden
    compute2 = kt.Compute(gpus="1")
    compute2.autoscale(
        target=10,
        scale_down_delay="5m",  # Override default
        scale_to_zero_pod_retention_period="30s",  # Override default
        progress_deadline="5m",  # Override default
        concurrency=5,  # Explicitly set
        min_scale=0,  # Explicitly set
    )

    assert compute2.autoscaling_config.scale_down_delay == "5m"
    assert compute2.autoscaling_config.scale_to_zero_pod_retention_period == "30s"
    assert compute2.autoscaling_config.progress_deadline == "5m"
    assert compute2.autoscaling_config.concurrency == 5
    assert compute2.autoscaling_config.min_scale == 0


@pytest.mark.level("unit")
def test_extra_annotations():
    """Test that extra kwargs become extra annotations"""
    config = AutoscalingConfig(
        min_scale=1,
        some_custom_annotation="custom-value",
        another_annotation="another-value",
    )
    annotations = config.convert_to_annotations()

    assert annotations["autoscaling.knative.dev/min-scale"] == "1"
    assert annotations["autoscaling.knative.dev/some_custom_annotation"] == "custom-value"
    assert annotations["autoscaling.knative.dev/another_annotation"] == "another-value"
