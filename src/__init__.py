import os
import sys

# get current working directory
PROJECT_ROOT = os.getcwd()

# strip file path to the root directory of the project
PROJECT_ROOT = PROJECT_ROOT.split(sep='/src')[0]
PROJECT_ROOT = PROJECT_ROOT.split(sep='/tests')[0]
PROJECT_ROOT = PROJECT_ROOT.split(sep='/notebooks')[0]
PROJECT_ROOT = PROJECT_ROOT.split(sep='/models')[0]
PROJECT_ROOT = PROJECT_ROOT.split(sep='/app')[0]

# add path where scripts are
SOURCE_PATH = os.path.join(
    PROJECT_ROOT, "src"
)

# add paths to the project structure
sys.path.append(PROJECT_ROOT)
sys.path.append(SOURCE_PATH)

# print('\nProject root from src/__init__: {}'.format(PROJECT_ROOT))
# print('\nSource path src/__init__: {}'.format(SOURCE_PATH))
