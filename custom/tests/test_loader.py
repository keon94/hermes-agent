import pytest

from custom_loader import get_runtime


def test_loader_is_opt_in_and_domain_runtime_is_resolved():
    assert get_runtime({}) is None
    runtime = get_runtime({'custom_runtime': 'job-finder'})
    assert runtime.handles({'custom_runtime': 'job-finder'})


def test_loader_rejects_unknown_runtime():
    with pytest.raises(ValueError, match='Unknown custom runtime'):
        get_runtime({'custom_runtime': 'not-installed'})
