import subprocess
import textwrap
import uuid

from functools import cached_property
from typing import Dict

from kubetorch import globals

from kubetorch.constants import DEFAULT_VOLUME_ACCESS_MODE
from kubetorch.logger import get_logger
from kubetorch.utils import http_not_found

logger = get_logger(__name__)


class Volume:
    """
    Manages persistent storage for Kubetorch services and deployments.
    """

    def __init__(
        self,
        name: str,
        size: str,
        mount_path: str,
        storage_class: str = None,
        access_mode: str = None,
        namespace: str = None,
        volume_name: str = None,
    ):
        """
        Kubetorch Volume object, specifying persistent storage properties.

        Args:
            name (str): Name of the volume/PVC.
            size (str): Size of the volume.
            mount_path (str): Mount path for the volume.
            storage_class (str, optional): Storage class to use for the volume.
                Ignored if volume_name is specified.
            access_mode (str, optional): Access mode for the volume.
            namespace (str, optional): Namespace for the volume.
            volume_name (str, optional): Name of an existing PersistentVolume (PV) to bind to.
                When specified, creates a PVC that binds to this specific PV instead of
                using dynamic provisioning via a storage class.

        Example:

        .. code-block:: python

            import kubetorch as kt

            # Standard volume (ReadWriteOnce)
            kt.Volume(name="my-data", size="5Gi", mount_path="/data")

            # Shared volume (ReadWriteMany, requires JuiceFS or similar)
            kt.Volume(
                name="shared-data",
                size="10Gi",
                mount_path="/shared",
                storage_class="juicefs-sc-shared",
                access_mode="ReadWriteMany"
            )

            # Bind to an existing PV
            kt.Volume(
                name="team-nfs-pvc",
                size="20Gi",
                mount_path="/data",
                volume_name="team-nfs-pv",
                access_mode="ReadWriteMany"
            )

            # uv cache
            compute = kt.Compute(
                cpus=".01",
                env_vars={
                    "UV_CACHE_DIR": "/cache/uv_cache",
                    "HF_HOME": "/cache/hf_cache",
                },
                volumes=[kt.Volume(name="kt-global-cache", size="10Gi", mount_path="/cache")],
            )

        """
        self._storage_class = storage_class
        self.size = size
        self.access_mode = access_mode or DEFAULT_VOLUME_ACCESS_MODE
        self.mount_path = mount_path
        self.volume_name = volume_name

        self.name = name
        self.namespace = namespace or globals.config.namespace
        self.controller_client = globals.controller_client()

    @property
    def pvc_name(self) -> str:
        return self.name

    @property
    def mount_path(self) -> str:
        """Get the mount path for this volume"""
        return self._mount_path

    @mount_path.setter
    def mount_path(self, value: str):
        """Set the mount path with validation"""
        if not value:
            raise ValueError("mount_path cannot be empty")
        if not value.startswith("/"):
            raise ValueError(f"mount_path must be an absolute path starting with '/', got: {value}")
        self._mount_path = value

    @cached_property
    def storage_class(self) -> str:
        """Get storage class - either specified or cluster default"""
        # When binding to an existing PV, storage class should be empty
        if self.volume_name:
            return ""

        if self._storage_class:
            return self._storage_class

        try:
            result = self.controller_client.list_storage_classes()
            storage_classes = result.get("items", [])

            # If RWX is requested, prefer RWX-capable classes
            if self.access_mode == "ReadWriteMany":
                for sc in storage_classes:
                    provisioner = sc.get("provisioner", "")
                    if provisioner in {
                        "csi.juicefs.com",
                        "nfs.csi.k8s.io",
                        "cephfs.csi.ceph.com",
                    }:
                        return sc["metadata"]["name"]
                raise ValueError("No RWX-capable storage class found")

            # Otherwise, pick the default StorageClass
            for sc in storage_classes:
                annotations = sc.get("metadata", {}).get("annotations") or {}
                if annotations.get("storageclass.kubernetes.io/is-default-class") == "true":
                    sc_name = sc["metadata"]["name"]
                    logger.info(f"Using default storage class: {sc_name}")
                    return sc_name

            # No default found, fall back to first available
            available_classes = [sc["metadata"]["name"] for sc in storage_classes]
            first_sc = available_classes[0]
            if len(available_classes) == 1:
                logger.info(f"No default storage class found, using only available one: {first_sc}")
            else:
                logger.warning(
                    f"No default storage class found, using first available: {first_sc}. "
                    f"Available: {available_classes}. Consider setting a default or specifying storage_class parameter."
                )
            return first_sc

        except Exception as e:
            logger.error(f"Failed to get storage classes: {e}")
            raise

    @classmethod
    def from_name(
        cls,
        name: str,
        namespace: str = None,
        mount_path: str = None,
    ) -> "Volume":
        """Get existing volume by name

        Args:
            name (str): Name of the volume/PVC
            namespace (str, optional): Kubernetes namespace
            mount_path (str, optional): Override the mount path. If not provided, uses the annotation from the PVC

        Returns:
            Volume: Volume instance loaded from existing PVC
        """
        controller_client = globals.controller_client()
        namespace = namespace or globals.config.namespace
        pvc_name = name

        try:
            pvc = controller_client.get_pvc(namespace, pvc_name)

            storage_class = pvc["spec"].get("storageClassName")
            size = pvc["spec"]["resources"]["requests"]["storage"]
            access_modes = pvc["spec"].get("accessModes", [])
            access_mode = access_modes[0] if access_modes else DEFAULT_VOLUME_ACCESS_MODE
            volume_name = pvc["spec"].get("volumeName")

            # Load mount_path from annotation
            annotations = pvc.get("metadata", {}).get("annotations") or {}
            annotation_mount_path = annotations.get("kubetorch.com/mount-path")

            # Use provided mount_path or fall back to annotation
            final_mount_path = mount_path if mount_path is not None else annotation_mount_path

            # Create Volume with actual attributes from PVC
            vol = cls(
                name=name,
                storage_class=storage_class,
                mount_path=final_mount_path,
                size=size,
                access_mode=access_mode,
                namespace=namespace,
                volume_name=volume_name,
            )

            if volume_name:
                logger.debug(f"Loaded existing PVC {pvc_name} bound to PV {volume_name}")
            else:
                logger.debug(f"Loaded existing PVC {pvc_name} with storage_class={storage_class}")
            return vol

        except Exception as e:
            if http_not_found(e):
                raise ValueError(f"Volume '{name}' (PVC: {pvc_name}) does not exist in namespace '{namespace}'")
            raise

    def config(self) -> Dict[str, str]:
        """Get configuration for this volume"""
        config = {
            "name": self.name,
            "size": self.size,
            "access_mode": self.access_mode,
            "mount_path": self.mount_path,
            "namespace": self.namespace,
        }
        if self.volume_name:
            config["volume_name"] = self.volume_name
        else:
            config["storage_class"] = self.storage_class
        return config

    def pod_template_spec(self) -> dict:
        """Convert to Kubernetes volume spec for pod template"""
        return {
            "name": self.name,
            "persistentVolumeClaim": {"claimName": self.pvc_name},
        }

    def create(self) -> Dict:
        """Create PVC if it doesn't exist"""
        try:
            # Check if PVC already exists
            existing_pvc = self.controller_client.get_pvc(self.namespace, self.pvc_name, ignore_not_found=True)
            logger.debug(f"PVC {self.pvc_name} already exists in namespace {self.namespace}")
            if existing_pvc:
                logger.debug(f"PVC {self.pvc_name} already exists in namespace {self.namespace}")
                return existing_pvc

            logger.info(f"Creating new PVC with name: {self.pvc_name}")

            # When binding to an existing PV, use empty storage class and set volumeName
            if self.volume_name:
                storage_class_name = ""
            else:
                storage_class_name = self.storage_class

            pvc_body = {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {
                    "name": self.pvc_name,
                    "labels": {
                        "app": "kubetorch",
                        "kubetorch.com/volume": self.name,
                    },
                    "annotations": {"kubetorch.com/mount-path": self.mount_path},
                },
                "spec": {
                    "accessModes": [self.access_mode],
                    "resources": {"requests": {"storage": self.size}},
                    "storageClassName": storage_class_name,
                },
            }

            # Add volumeName to bind to a specific PV
            if self.volume_name:
                pvc_body["spec"]["volumeName"] = self.volume_name

            created_pvc = self.controller_client.create_pvc(self.namespace, pvc_body)

            if self.volume_name:
                logger.info(
                    f"Successfully created PVC {self.pvc_name} in namespace {self.namespace} "
                    f"bound to PV {self.volume_name}"
                )
            else:
                logger.info(
                    f"Successfully created PVC {self.pvc_name} in namespace {self.namespace} with "
                    f"storage class {storage_class_name}"
                )
            return created_pvc

        except Exception as e:
            logger.error(f"Failed to create PVC {self.pvc_name}: {e}")
            raise

    def delete(self, wait: bool = True, timeout: int = 60) -> None:
        """Delete the PVC and optionally wait for deletion to complete.

        Args:
            wait: Whether to wait for the PVC to be fully deleted (default: True)
            timeout: Maximum time to wait for deletion in seconds (default: 60)
        """
        import time

        try:
            self.controller_client.delete_pvc(self.namespace, self.pvc_name)
            logger.debug(f"Initiated deletion of PVC {self.pvc_name}")

            if wait:
                # Wait for PVC to be fully deleted
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if not self.exists():
                        logger.debug(f"Successfully deleted PVC {self.pvc_name}")
                        return
                    time.sleep(0.5)

                # Timeout - PVC still exists
                logger.warning(f"PVC {self.pvc_name} deletion timed out after {timeout}s (may still be terminating)")

        except Exception as e:
            if http_not_found(e):
                logger.warning(f"PVC {self.pvc_name} not found")
                return
            logger.error(f"Failed to delete PVC {self.pvc_name}: {e}")
            raise

    def exists(self) -> bool:
        """Check if the PVC exists"""
        try:
            self.controller_client.get_pvc(self.namespace, self.pvc_name)
            return True
        except Exception as e:
            if http_not_found(e):
                return False
            raise

    def ssh(self, image: str = "alpine:latest"):
        """
        Launch an interactive debug shell with this volume mounted.

        This method creates a temporary Kubernetes pod that mounts the
        PersistentVolumeClaim (PVC) backing this Volume at the same path
        (`self.mount_path`) used by Kubetorch services.

        Args:
            image (str, optional): Container image to use for the debug pod.
                Must include a shell (e.g., `alpine:3.18`, `ubuntu:22.04`,
                or a custom tools image). Defaults to `alpine:latest`.

        Example:

        .. code-block:: python

            import kubetorch as kt

            vol = kt.Volume.from_name("kt-global-cache")
            vol.ssh()
        """
        pod_name = f"debug-{self.name}-{uuid.uuid4().hex[:6]}"

        cmd = [
            "kubectl",
            "run",
            pod_name,
            "--rm",
            "-it",
            "--namespace",
            self.namespace,
            "--image",
            image,
            "--restart=Never",
            "--overrides",
            textwrap.dedent(
                f"""
            {{
              "apiVersion": "v1",
              "spec": {{
                "containers": [{{
                  "name": "debug",
                  "image": "{image}",
                  "stdin": true,
                  "tty": true,
                  "volumeMounts": [{{
                    "name": "vol",
                    "mountPath": "{self.mount_path}"
                  }}]
                }}],
                "volumes": [{{
                  "name": "vol",
                  "persistentVolumeClaim": {{
                    "claimName": "{self.pvc_name}"
                  }}
                }}]
              }}
            }}
            """
            ).strip(),
        ]

        # Suppress noisy "write on closed stream" error when exiting
        subprocess.run(cmd, stderr=subprocess.DEVNULL)
