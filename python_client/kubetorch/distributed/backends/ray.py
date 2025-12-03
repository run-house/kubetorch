"""Ray distributed backend."""

import multiprocessing
import os
import subprocess
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Dict, Optional

import httpx

from kubetorch.distributed.base import DistributedProcess, DistributedSupervisor
from kubetorch.distributed.http_worker_pool import RemoteWorkerPool
from kubetorch.distributed.process_pool import DistributedProcessPool
from kubetorch.logger import get_logger
from kubetorch.servers.http.http_server import patch_sys_path

logger = get_logger(__name__)

RAY_START_PROC = None


class RayProcess(DistributedProcess):
    """Ray-specific distributed process."""

    def proc_cleanup(self):
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown completed.")
        except ImportError:
            logger.info("Ray not available for cleanup")
        except Exception as e:
            logger.info(f"Failed to shutdown Ray: {e}")
        # Call parent cleanup for debugging sessions
        super().proc_cleanup()


class RayDistributed(DistributedSupervisor):
    def __init__(
        self,
        restart_procs=True,
        max_threads=4,
        quorum_timeout=300,
        quorum_workers=None,
        monitor_members=False,
    ):
        """Ray distributed supervisor - only runs on head node (single controller).

        Args:
            restart_procs: Whether to restart processes on each call
            max_threads: Maximum threads per process
            quorum_timeout: Timeout in seconds for Ray cluster nodes to become ready (default 300s/5min)
        """
        # Ray manages its own membership, so we don't monitor DNS changes
        super().__init__(
            quorum_timeout=quorum_timeout,
            quorum_workers=quorum_workers,
            monitor_members=monitor_members,
        )
        self.restart_procs = restart_procs
        self.distributed_env_vars = None
        self.process_pool = None  # Using pool even for single process for consistency
        self.remote_worker_pool = None  # Pool for async HTTP calls to remote workers
        self.max_threads = max_threads
        self.quorum_timeout = quorum_timeout

    def setup(self, deployed_as_of: Optional[str] = None):
        # Set multiprocessing to spawn if not already
        if multiprocessing.get_start_method() != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

        # Start the Ray server here, if we allow KubeRay to start it in the pod template
        # it's hard to wait for it start properly and we lose the ability to restart if needed.
        global RAY_START_PROC

        # Check if Ray is actually running, not just if our global variable is None
        # (the global variable gets reset when HTTP server restarts)
        ray_running = self._is_ray_running()

        if not ray_running:
            patch_sys_path()

            kuberay_start_cmd = os.environ.get("KUBERAY_GEN_RAY_START_CMD")
            if kuberay_start_cmd:
                full_cmd = f"ulimit -n 65536; {kuberay_start_cmd}"
                logger.info(f"Starting Ray server with command: {full_cmd}")

                try:
                    # Start Ray as a non-blocking subprocess
                    RAY_START_PROC = subprocess.Popen(
                        full_cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1,
                        env=os.environ.copy(),
                    )

                    # Start a thread to stream Ray logs
                    def stream_ray_logs():
                        try:
                            for line in RAY_START_PROC.stdout:
                                logger.info(f"[Ray] {line.strip()}")
                        except Exception as e:
                            logger.error(f"Error streaming Ray logs: {e}")

                    log_thread = threading.Thread(target=stream_ray_logs, daemon=True)
                    log_thread.start()

                    logger.info(f"Ray server started with PID: {RAY_START_PROC.pid}")

                    # Give Ray a moment to start
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Failed to start Ray server: {e}")
                    RAY_START_PROC = None
                    raise
            else:
                logger.warning("KUBERAY_GEN_RAY_START_CMD environment variable not found")

        logger.debug("Ray distributed supervisor setup completed (pod discovery will be done lazily)")

        # Only the head node runs the subprocess
        this_pod_ip = os.environ["POD_IP"]
        if not os.environ["POD_NAME"].endswith("-head"):
            logger.info(f"Ray worker node {this_pod_ip}, skipping subprocess setup")
            return

        logger.info(f"Ray head node {this_pod_ip}, setting up subprocess")

        # Set Ray environment variables
        self.distributed_env_vars = {"RAY_HEAD_NODE_IP": this_pod_ip}

        # Include critical environment variables so Ray workers can find and load the callable
        critical_env_vars = [
            "PYTHONPATH",
            "KT_FILE_PATH",
            "KT_MODULE_NAME",
            "KT_CLS_OR_FN_NAME",
        ]
        for env_var in critical_env_vars:
            if env_var in os.environ:
                self.distributed_env_vars[env_var] = os.environ[env_var]

        # Cleanup will remove the process pool if found, so we need to check if it was previously initialized
        previously_initialized = self.remote_worker_pool is not None

        if self.restart_procs:
            logger.debug("restart_procs is True, restarting Ray distributed process")
            self.cleanup()

            if previously_initialized:
                pod_ips = self.pod_ips()
                this_pod_ip = os.environ["POD_IP"]

                # Send reload requests to other pods if needed
                self._reload_image_on_other_pods(pod_ips, this_pod_ip, deployed_as_of)

        if self.process_pool is None:
            logger.debug("Setting up Ray distributed process")
            self.process_pool = DistributedProcessPool(
                process_class=RayProcess,
                num_processes=1,  # Ray only needs one process
                max_threads_per_proc=self.max_threads,
            )
            self.process_pool.start()

            # # Start remote worker pool for async HTTP calls if needed
            # Use a reasonable default max_workers since we don't know cluster size yet
            self.remote_worker_pool = RemoteWorkerPool(quorum_timeout=self.quorum_timeout)
            self.remote_worker_pool.start(max_workers=100)  # Default size

            logger.debug("Finished setting up Ray distributed process and remote worker pool")

    def cleanup(self):
        """Clean up Ray distributed process."""
        logger.debug("Cleaning up Ray distributed process")

        # Stop DNS monitoring first
        super().cleanup()

        if self.process_pool:
            self.process_pool.stop()
            self.process_pool = None

        if self.remote_worker_pool:
            self.remote_worker_pool.stop()
            self.remote_worker_pool = None

        logger.debug("Finished cleaning up Ray distributed process")

    @staticmethod
    def intercept_call():
        return True

    def call_distributed(
        self,
        request,
        cls_or_fn_name: str,
        method_name: Optional[str] = None,
        params: Optional[Dict] = None,
        distributed_subcall: bool = False,
        debug_port: int = False,
        deployed_as_of: Optional[str] = None,
    ):
        """Ray distributed call - only executes on head node."""
        request_id = request.headers.get("X-Request-ID", "-")
        serialization = request.headers.get("X-Serialization", "json")

        # If deployed_as_of is None, generate a consistent timestamp
        # to use across all workers to prevent reload inconsistencies
        if deployed_as_of is None:
            from datetime import datetime, timezone

            deployed_as_of = datetime.now(timezone.utc).isoformat()

        if not os.environ["POD_NAME"].endswith("-head"):
            # This should never happen, because the service only points to the head node, Raise an error if it does.
            raise RuntimeError(
                f"Ray distributed call attempted on non-head node {os.environ['POD_NAME']}. "
                "This should only be called on the head node."
            )

        # Start DNS monitoring for the head node
        self.start_dns_monitoring()

        # Check for any pending changes before we start
        self.check_for_membership_changes()

        # The pod_ips() method now handles waiting for quorum
        pod_ips = self.pod_ips()

        # Handle case where no pods are found
        if not pod_ips:
            logger.error(
                f"No pods found for service {os.environ.get('KT_SERVICE')}. "
                "This may indicate the pods aren't ready yet. Consider increasing quorum_timeout in .distribute() call."
            )
            raise RuntimeError(
                "No pods found for Ray distributed setup. " "Consider increasing quorum_timeout parameter."
            )

        logger.info(f"Found {len(pod_ips)} pod(s) for distributed setup: {pod_ips}")

        # Update distributed env vars with current cluster IPs
        self.distributed_env_vars["POD_IPS"] = ",".join(pod_ips)

        logger.debug("Sending call to Ray subprocess")
        # Ray uses only one process, so always call index 0
        result = self.process_pool.call(
            idx=0,
            method_name=method_name,
            params=params,
            deployed_as_of=deployed_as_of,
            request_id=request_id,
            distributed_env_vars=self.distributed_env_vars,
            debug_port=debug_port,
            serialization=serialization,
        )

        # Handle exceptions from subprocess
        try:
            from starlette.responses import JSONResponse

            is_json_response = isinstance(result, JSONResponse)
        except ImportError:
            is_json_response = False
        if is_json_response:
            return result
        if isinstance(result, Exception):
            raise result

        return result

    def _reload_image_on_other_pods(self, pod_ips, this_pod_ip, deployed_as_of):
        """Send /_reload_image requests to all other pods in parallel, with retries for pods that aren't ready."""
        other_pod_ips = [ip for ip in pod_ips if ip != this_pod_ip]

        if not other_pod_ips:
            logger.debug("No other pods to reload")
            return

        logger.info(f"Sending reload requests to {len(other_pod_ips)} other pods: {other_pod_ips}")

        server_port = os.environ.get("KT_SERVER_PORT", "32300")
        total_timeout = self.quorum_timeout  # Use configurable quorum timeout
        retry_interval = 2  # Wait 2 seconds between retry attempts
        start_time = time.time()

        successful_pods = set()
        remaining_pods = set(other_pod_ips)

        while remaining_pods and (time.time() - start_time) < total_timeout:
            logger.debug(f"Attempting to reload {len(remaining_pods)} remaining pods: {list(remaining_pods)}")

            def reload_pod(pod_ip):
                """Send reload request to a single pod."""
                try:
                    # Use a proper HTTP client session to avoid Content-Length issues
                    with httpx.Client(timeout=None) as client:
                        url = f"http://{pod_ip}:{server_port}/_reload_image"
                        # First try a quick health check to see if pod is ready
                        health_url = f"http://{pod_ip}:{server_port}/health"
                        health_response = client.get(health_url, timeout=5)

                        if health_response.status_code != 200:
                            logger.debug(f"Pod {pod_ip} health check failed, will retry later")
                            return False

                        # Pod is healthy, send reload request (no timeout, installs can be long-running)
                        response = client.post(url, headers={"X-Deployed-As-Of": deployed_as_of})
                        if response.status_code == 200:
                            logger.debug(f"Successfully reloaded image on pod {pod_ip}")
                            return True
                        else:
                            logger.warning(f"Pod {pod_ip} reload returned status {response.status_code}")
                            return False

                except Exception as e:
                    logger.debug(f"Failed to reload image on pod {pod_ip}: {e}")
                    raise

            # Try to reload all remaining pods in parallel
            current_attempt_pods = list(remaining_pods)

            with ThreadPoolExecutor(max_workers=min(len(current_attempt_pods), 10)) as executor:
                # Submit reload tasks for remaining pods
                future_to_pod = {executor.submit(reload_pod, pod_ip): pod_ip for pod_ip in current_attempt_pods}

                # Process completed futures
                for future in as_completed(future_to_pod, timeout=None):
                    pod_ip = future_to_pod[future]
                    try:
                        success = future.result()
                        if success:
                            successful_pods.add(pod_ip)
                            remaining_pods.discard(pod_ip)
                    except Exception as e:
                        logger.debug(f"Reload task for pod {pod_ip} failed: {e}")

            if remaining_pods:
                elapsed = time.time() - start_time
                remaining_time = total_timeout - elapsed
                if remaining_time > retry_interval:
                    logger.info(f"Waiting {retry_interval}s before retrying {len(remaining_pods)} pods...")
                    time.sleep(retry_interval)
                else:
                    logger.warning("Timeout approaching, stopping retry attempts")
                    break

        # Log final results
        if successful_pods:
            logger.info(f"Successfully reloaded {len(successful_pods)} pod images: {list(successful_pods)}")

        if remaining_pods:
            logger.warning(f"Failed to reload {len(remaining_pods)} pod images after timeout: {list(remaining_pods)}")

    def _is_ray_running(self):
        """Check if Ray is actually running by trying to connect to the Ray GCS port."""
        try:
            import socket

            # Ray GCS runs on port 6379 by default
            ray_port = 6379
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)  # 1 second timeout
            result = sock.connect_ex(("127.0.0.1", ray_port))
            sock.close()

            if result == 0:
                logger.debug("Ray GCS port 6379 is accessible, Ray appears to be running")
                return True
            else:
                logger.debug("Ray GCS port 6379 is not accessible, Ray is not running")
                return False

        except Exception as e:
            logger.debug(f"Error checking if Ray is running: {e}")
            return False
