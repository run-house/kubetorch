"""
Test script for the metadata server.

Run this to verify the metadata server is working correctly.
"""


import requests

BASE_URL = "http://localhost:8081"


def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    print("✓ Health check passed")


def test_publish():
    """Test publishing a key."""
    response = requests.post(
        f"{BASE_URL}/api/v1/keys/test-key/publish",
        json={"ip": "10.244.1.5"},
    )
    assert response.status_code == 200
    print("✓ Publish test-key from 10.244.1.5 passed")


def test_publish_multiple():
    """Test publishing from multiple pods."""
    for i in range(3):
        response = requests.post(
            f"{BASE_URL}/api/v1/keys/test-key/publish",
            json={"ip": f"10.244.1.{5 + i}"},
        )
        assert response.status_code == 200
    print("✓ Published from multiple pods")


def test_get_source():
    """Test getting source IP."""
    response = requests.get(f"{BASE_URL}/api/v1/keys/test-key/source")
    assert response.status_code == 200
    data = response.json()
    assert "ip" in data
    print(f"✓ Got source IP: {data['ip']}")


def test_register_store():
    """Test registering store pod."""
    response = requests.post(
        f"{BASE_URL}/api/v1/keys/test-key/store",
        json={"ip": "10.244.1.3"},
    )
    assert response.status_code == 200
    print("✓ Registered store pod")


def test_get_source_prefers_store():
    """Test that store pod is preferred."""
    store_count = 0
    for _ in range(20):
        response = requests.get(f"{BASE_URL}/api/v1/keys/test-key/source")
        data = response.json()
        if data["ip"] == "10.244.1.3":
            store_count += 1
    # Should get store pod ~80% of the time (16 out of 20)
    assert store_count >= 12, f"Expected store pod preference, got {store_count}/20"
    print(
        f"✓ Store pod preference works ({store_count}/20 requests returned store pod)"
    )


def test_remove_source():
    """Test removing a source."""
    response = requests.delete(f"{BASE_URL}/api/v1/keys/test-key/sources/10.244.1.5")
    assert response.status_code == 200
    print("✓ Removed source IP")


def test_get_key_info():
    """Test getting key info."""
    response = requests.get(f"{BASE_URL}/api/v1/keys/test-key")
    assert response.status_code == 200
    data = response.json()
    assert "sources" in data
    assert "store_pod_ip" in data
    print(
        f"✓ Got key info: {len(data['sources'])} sources, store_pod_ip={data['store_pod_ip']}"
    )


def test_get_stats():
    """Test getting stats."""
    response = requests.get(f"{BASE_URL}/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_keys" in data
    print(f"✓ Got stats: {data}")


def test_nonexistent_key():
    """Test getting source for nonexistent key."""
    response = requests.get(f"{BASE_URL}/api/v1/keys/nonexistent-key/source")
    assert response.status_code == 200
    data = response.json()
    assert data.get("found") is False
    print("✓ Nonexistent key returns found=false")


if __name__ == "__main__":
    print("Testing metadata server...")
    print(f"Base URL: {BASE_URL}\n")

    try:
        test_health()
        test_publish()
        test_publish_multiple()
        test_get_source()
        test_register_store()
        test_get_source_prefers_store()
        test_remove_source()
        test_get_key_info()
        test_get_stats()
        test_nonexistent_key()

        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
