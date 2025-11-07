import os
from fnmatch import fnmatch
import numpy as np

#path
nbuddies_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/"

#subdirectories of src
code_subdirectories = np.array([
    "tests"
])

#Output directories
output_directories = np.array([
    "data",
    "movie_dump",
    "visuals"
])

#Required sundirectories of nbuddies
required_directories = np.concatenate((output_directories, [
    "src"
], "src/" + code_subdirectories))

#Authorized subdirectories of nbuddies
authorized_directories = np.append(required_directories, [
    ".*", #system and github hidden files and directories
    "__*", #python files
    "data/*",
    "movie_dump/*",
    "visuals/*"
])

def test_organization():
    """
    Checks if file tree matches authorized file tree, creates needed directories if they don't exist
    """
    #check and create required directories
    for directory in required_directories:
        if not os.path.exists(nbuddies_path + directory):
            os.mkdir(nbuddies_path + directory)

    #check for forbidden directories
    for suspect in os.listdir(nbuddies_path):
        if os.path.isfile(nbuddies_path + suspect):
            assert fnmatch(suspect, "*.py") or fnmatch(suspect, ".*") or fnmatch(suspect, "LICENSE"), f"Data file {suspect} found outside of data directory! Contain it!!"

        else:
            allowed = False
            for directory in authorized_directories:
                if fnmatch(suspect, directory):
                    allowed = True

            assert allowed, f"Unauthorized directory {suspect} detected in nbuddies"


def test_output_directories():
    #There shall be no files in output directories
    for directory in output_directories:
        for suspect in os.listdir(nbuddies_path + directory):
            assert os.path.isdir(nbuddies_path + directory + "/" + suspect), f"Forbidden file {suspect} in {directory}! Please contain to a subdirectory"

def test_src():
    #check for forbidden directories
    for suspect in os.listdir(nbuddies_path + "src"):
        if os.path.isfile(nbuddies_path + "src/" + suspect):
            assert fnmatch(suspect, "*.py"), f"Non-python file {suspect} detected in src!"
        
        else:
            allowed = False
            for directory in code_subdirectories:
                if fnmatch(suspect, directory) or fnmatch(suspect, ".*") or fnmatch(suspect, "__*"):
                    allowed = True

            assert allowed, f"Unauthorized directory {suspect} detected in src"
