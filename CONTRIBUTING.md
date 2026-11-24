# Contributing to insight-vio

Thanks for your interest! Bug reports, fixes, docs and hardware results are all welcome.

1. Open an issue first for anything bigger than a small fix, so we can agree on the approach.
2. Fork, create a branch, and keep each pull request focused on one change.
3. Run the tests before pushing:
   ```bash
   python -m pytest -q
   ```
4. Add or update tests for behaviour changes, and keep the README honest: only report numbers
   you measured, and say on what machine or hardware.
5. Code style: PEP 8, type hints where they help, 4-space indentation (see `.editorconfig`).

Hardware results are especially valuable. Please include the exact command, config file,
hardware and commit hash so others can reproduce them.

By contributing you agree that your contributions are licensed under the MIT License.
