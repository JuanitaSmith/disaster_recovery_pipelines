# import os

# sys.path.append(config.folder_scripts)

# sys.path.append('../')

import os
import sys

# get current working directory
PROJECT_PATH = os.getcwd()

# remove the current 'test' directory
PROJECT_PATH = PROJECT_PATH.split(sep='/test')[0]

# add path where scripts are
SOURCE_PATH = os.path.join(
    PROJECT_PATH,"src"
)

sys.path.append(SOURCE_PATH)

print('\nProject path: {}'.format(PROJECT_PATH))
print('\nSource path: {}'.format(SOURCE_PATH))
