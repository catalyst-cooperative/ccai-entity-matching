"""A dummy unit test so pytest has something to do."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def test_nothing(test_dir: Path) -> None:
    """A dummy test that relies on our dummy fixture."""
    assert isinstance(test_dir, Path)  # nosec: B101
    assert test_dir.exists()  # nosec: B101
    assert test_dir.is_dir()  # nosec: B101
