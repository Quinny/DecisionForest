# C++ Decision Forest

A fast, multi-threaded Decision Forest C++ library which supports user defined splitting functions.  For example usage, see main.cpp.

## Build Instructions

This library does not have any external dependencies, and requires a C++14 complaint compiler.

### Sharcnet Build

1. Load the proper modules: `./load_modules.sh`

2. Build the code: `make sharc`

3. Submit the job to the queue: `./submit_job.sh <runtime> <memory> <output_file> <executable>`

Note that with a large number of trees and the mnist dataset, the program generally
requires ~3GB.  For runtime approximations see the results section of the accompanying
paper.
