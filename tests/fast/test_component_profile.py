from miles.utils import component_profile


def test_profile_each_call_numbers_ranges(monkeypatch):
    ranges = []

    class RecordRange:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            ranges.append(self.name)

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(component_profile, "_ENABLED", True)
    monkeypatch.setattr(component_profile.torch.profiler, "record_function", RecordRange)

    @component_profile.profile_each_call("microbatch")
    def identity(value):
        return value

    assert identity("first") == "first"
    assert identity("second") == "second"
    assert ranges == ["microbatch.0", "microbatch.1"]
