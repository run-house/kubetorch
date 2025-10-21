import subprocess
import textwrap
import uuid

from functools import cached_property
from typing import Dict

from kubernetes import client
from kubernetes.client import ApiException, V1PersistentVolumeClaim

from kubetorch.constants import DEFAULT_VOLUME_ACCESS_MODE, KT_MOUNT_FOLDER
from kubetorch.globals import config
from kubetorch.logger import get_logger
from kubetorch.utils import load_kubeconfig

logger = get_logger(__name__)


class Volume:
    """
    Manages persistent storage for Kubetorch services and deployments.
    """

    def __init__(
        self,
        name: str,
        size: str,
        storage_class: str = None,
        mount_path: str = None,
        access_mode: str = None,
        namespace: str = None,
        core_v1: client.CoreV1Api = None,
    ):
        """
        Kubetorch Volume object, specifying persistent storage properties.

        Args:
            name (str): Name of the volume.
            size (str): Size of the volume.
            storage_class (str, optional): Storage class to use for the volume.
            mount_path (str, optional): Mount path for the volume.
            access_mode (str, optional): Access mode for the volume.
            namespace (str, optional): Namespace for the volume.

        Example:

        .. code-block:: python

            import kubetorch as kt

            kt.Volume(name="my-data", size="5Gi"),  # Standard volume (ReadWriteOnce)

            # Shared volume (ReadWriteMany, requires JuiceFS or similar)
            kt.Volume(name="shared-data", size="10Gi", storage_class="juicefs-sc-shared", access_mode="ReadWriteMany")

            # uv cache
            compute = kt.Compute(
                cpus=".01",
                env_vars={
                    "UV_CACHE_DIR": "/ktfs/kt-global-cache/uv_cache",
                    "HF_HOME": "/ktfs/kt-global-cache/hf_cache",
                },
                volumes=[kt.Volume("kt-global-cache", size="10Gi")],
            )

        """
        self._storage_class = storage_class
        if core_v1 is None:
            load_kubeconfig()

        self.size = size
        self.access_mode = access_mode or DEFAULT_VOLUME_ACCESS_MODE
        self.mount_path = mount_path or f"/{KT_MOUNT_FOLDER}/{name}"

        self.name = name
        self.namespace = namespace
        self.core_v1 = core_v1 or client.CoreV1Api()

    @property
    def pvc_name(self) -> str:
        return self.name

    @cached_property
    def storage_class(self) -> str:
        """Get storage class - either specified or cluster default"""
        if self._storage_class:
            return self._storage_class

        try:
            storage_v1 = client.StorageV1Api()
            storage_classes = storage_v1.list_storage_class().items

            # If RWX is requested, prefer RWX-capable classes
            if self.access_mode == "ReadWriteMany":
                for sc in storage_classes:
                    provisioner = getattr(sc, "provisioner", "")
                    if provisioner in {
                        "csi.juicefs.com",
                        "nfs.csi.k8s.io",
                        "cephfs.csi.ceph.com",
                    }:
                        return sc.metadata.name
                raise ValueError("No RWX-capable storage class found")

            # Otherwise, pick the default StorageClass
            for sc in storage_classes:
                annotations = sc.metadata.annotations or {}
                if (
                    annotations.get("storageclass.kubernetes.io/is-default-class")
                    == "true"
                ):
                    logger.info(f"Using default storage class: {sc.metadata.name}")
                    return sc.metadata.name

            # No default found, fall back to first available
            available_classes = [sc.metadata.name for sc in storage_classes]
            first_sc = available_classes[0]
            if len(available_classes) == 1:
                logger.info(
                    f"No default storage class found, using only available one: {first_sc}"
                )
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
        create_if_missing: bool = False,
        namespace: str = None,
        core_v1: client.CoreV1Api = None,
    ) -> "Volume":
        """Get existing volume or optionally create it"""
        if core_v1 is None:
            load_kubeconfig()
            core_v1 = client.CoreV1Api()

        namespace = namespace or config.namespace
        pvc_name = name

        try:
            pvc = core_v1.read_namespaced_persistent_volume_claim(pvc_name, namespace)

            storage_class = pvc.spec.storage_class_name
            size = pvc.spec.resources.requests.get("storage")
            access_mode = (
                pvc.spec.access_modes[0]
                if pvc.spec.access_modes
                else DEFAULT_VOLUME_ACCESS_MODE
            )

            annotations = pvc.metadata.annotations or {}
            mount_path = annotations.get(
                "kubetorch.com/mount-path", f"/{KT_MOUNT_FOLDER}/{name}"
            )

            # Create Volume with actual attributes from PVC
            vol = cls(
                name=name,
                storage_class=storage_class,
                mount_path=mount_path,
                size=size,
                access_mode=access_mode,
                namespace=namespace,
                core_v1=core_v1,
            )

            logger.debug(
                f"Loaded existing PVC {pvc_name} with storage_class={storage_class}"
            )
            return vol

        except ApiException as e:
            if e.status == 404:
                # PVC doesn't exist
                if create_if_missing:
                    vol = cls(name, namespace=namespace, core_v1=core_v1)
                    vol.create()
                    return vol
                else:
                    raise ValueError(
                        f"Volume '{name}' (PVC: {pvc_name}) does not exist in namespace '{namespace}'"
                    )
            else:
                # Some other API error
                raise

    def config(self) -> Dict[str, str]:
        """Get configuration for this volume"""
        return {
            "name": self.name,
            "size": self.size,
            "access_mode": self.access_mode,
            "mount_path": self.mount_path,
            "storage_class": self.storage_class,
            "namespace": self.namespace,
        }

    def pod_template_spec(self) -> dict:
        """Convert to Kubernetes volume spec for pod template"""
        return {
            "name": self.name,
            "persistentVolumeClaim": {"claimName": self.pvc_name},
        }

    def create(self) -> V1PersistentVolumeClaim:
        """Create PVC if it doesn't exist"""
        try:
            try:
                # Check if PVC already exists
                existing_pvc = self.core_v1.read_namespaced_persistent_volume_claim(
                    name=self.pvc_name, namespace=self.namespace
                )
                logger.debug(
                    f"PVC {self.pvc_name} already exists in namespace {self.namespace}"
                )
                return existing_pvc
            except ApiException as e:
                if e.status != 404:
                    # Some other error occurred
                    raise

            logger.info(f"Creating new PVC with name: {self.pvc_name}")

            storage_class_name = self.storage_class

            pvc_spec = client.V1PersistentVolumeClaimSpec(
                access_modes=[self.access_mode],
                resources=client.V1ResourceRequirements(
                    requests={"storage": self.size}
                ),
                storage_class_name=storage_class_name,
            )

            pvc_metadata = client.V1ObjectMeta(
                name=self.pvc_name,
                labels={
                    "app": "kubetorch",
                    "kubetorch.com/volume": self.name,
                },
                annotations={"kubetorch.com/mount-path": self.mount_path},
            )

            pvc = client.V1PersistentVolumeClaim(
                api_version="v1",
                kind="PersistentVolumeClaim",
                metadata=pvc_metadata,
                spec=pvc_spec,
            )

            created_pvc = self.core_v1.create_namespaced_persistent_volume_claim(
                namespace=self.namespace, body=pvc
            )

            logger.info(
                f"Successfully created PVC {self.pvc_name} in namespace {self.namespace} with "
                f"storage class {storage_class_name}"
            )
            return created_pvc

        except Exception as e:
            logger.error(f"Failed to create PVC {self.pvc_name}: {e}")
            raise

    def delete(self) -> None:
        """Delete the PVC"""
        try:
            self.core_v1.delete_namespaced_persistent_volume_claim(
                name=self.pvc_name, namespace=self.namespace
            )
            logger.debug(f"Successfully deleted PVC {self.pvc_name}")
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"PVC {self.pvc_name} not found")
            else:
                logger.error(f"Failed to delete PVC {self.pvc_name}: {e}")
                raise

    def exists(self) -> bool:
        """Check if the PVC exists"""
        try:
            self.core_v1.read_namespaced_persistent_volume_claim(
                name=self.pvc_name, namespace=self.namespace
            )
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            else:
                # Some other API error, re-raise
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
