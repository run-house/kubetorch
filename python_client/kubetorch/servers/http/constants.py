# Profiling constants
PYSPY_SAMPLE_RATE_HZ = 100
SUPPORTED_PROFILERS = ["pytorch", "pyspy"]
SUPPORTED_PYSPY_OUTPUTS = ["flamegraph", "raw", "speedscope", "chrometrace"]
SUPPORTED_PYTORCH_OUTPUTS = ["chrometrace", "table", "memory_timeline", "stacks"]
SUPPORTED_PYTORCH_TABLE_SORT_KEYS = [
    "cpu_time",
    "cuda_time",
    "cpu_time_total",
    "cuda_time_total",
    "cpu_memory_usage",
    "cuda_memory_usage",
    "self_cpu_memory_usage",
    "self_cuda_memory_usage",
    "count",
]
SUPPORTED_PYTORCH_STACKS_METRICS = ["self_cpu_time_total", "self_cuda_time_total"]
SUPPORTED_PYTORCH_MEMORY_TIMELINE_OUTPUT = ["html", "json", "json_zip", "raw"]
