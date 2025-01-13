# This probably belongs in dallinger.db
# isort: off
import psycogreen.gevent

psycogreen.gevent.patch_psycopg()  # noqa
# isort: on

# This definitely belongs in dallinger.data
import psycopg2  # noqa
from contextlib import contextmanager  # noqa
from dallinger import data  # noqa

is_patched = getattr(data, "_patched", False)


@contextmanager
def _paused_thread():
    try:
        thread = psycopg2.extensions.get_wait_callback()
        psycopg2.extensions.set_wait_callback(None)
        yield
    finally:
        psycopg2.extensions.set_wait_callback(thread)


if not is_patched:
    _orig_copy_db_to_csv = data.copy_db_to_csv

    def _patched_copy_db_to_csv(dsn, path, scrub_pii=False):
        with _paused_thread():
            return _orig_copy_db_to_csv(dsn, path, scrub_pii)

    data._patched = True
    data.copy_db_to_csv = _patched_copy_db_to_csv
