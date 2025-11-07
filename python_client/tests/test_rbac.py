import pytest

from .utils import create_random_name_prefix, get_tests_namespace, simple_summer


@pytest.mark.level("minimal")
def test_fn_to_incorrect_namespace():
    import kubetorch as kt

    namespace_name = get_tests_namespace()
    print(f"Using namespace: {namespace_name}")

    fn_name = f"{create_random_name_prefix()}-summer-new-ns"

    with pytest.raises(kt.RsyncError) as e:
        kt.fn(simple_summer, name=fn_name).to(kt.Compute(cpus=".01", namespace=namespace_name, gpu_anti_affinity=True))
    assert e.value.returncode == 12  # rsync protocol data stream
    assert "Connection reset" in e.value.stderr
