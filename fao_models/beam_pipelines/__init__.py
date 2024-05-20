import sys
import os

# Get the parent directory path
parent_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python module search path
sys.path.append(parent_dir)
