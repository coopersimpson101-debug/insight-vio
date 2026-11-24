"""Make the repo importable when scripts are run without `pip install -e .`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
