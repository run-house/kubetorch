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
    assert e.value.returncode != 0  # rsync failed
    # Error message contains rsync protocol data stream error (code 12) or connection reset
    error_text = str(e.value) + e.value.stderr
    assert "code 12" in error_text or "Connection reset" in error_text
