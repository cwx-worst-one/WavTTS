from samantha.dataio.lite.utils.stats import UpdateStatsMixin


class Base(object):
    def __init__(self):
        pass


class Impl(Base, UpdateStatsMixin):
    def __init__(self):
        super().__init__()


def test_update_stats_mixin():
    update_stats = Impl()
    update_stats.log_interval = 1
    update_stats.update_stats(skipped=True, message="test")

    messages = update_stats.stats_proxy.messages
    assert len(messages) == 1
    assert messages["[Impl] test"] == 1
