import fnmatch
import os


dir_path = "/Users/myounes/Documents/Code/thesis/output"
count = len(fnmatch.filter(os.listdir(dir_path), '*.*'))
print(count)