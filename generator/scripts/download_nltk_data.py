from __future__ import annotations

import nltk


def main() -> None:
    for package in ("wordnet", "omw-1.4"):
        if not nltk.download(package, quiet=False):
            raise RuntimeError(f"Không tải được NLTK package: {package}")


if __name__ == "__main__":
    main()

