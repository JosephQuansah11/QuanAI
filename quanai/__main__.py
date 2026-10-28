"""``python -m quanai`` entry point."""

from .cli import main

if __name__ == "__main__":  # pragma: no cover - only true for ``python -m quanai``
    raise SystemExit(main())
