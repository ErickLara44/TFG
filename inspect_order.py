import sys
import os

sys.path.append(os.path.dirname(__file__))

from src.config import DEFAULT_VARIABLES

print("DEFAULT_VARIABLES Order:")
for i, k in enumerate(DEFAULT_VARIABLES.keys()):
    print(f"{i}: {k}")
