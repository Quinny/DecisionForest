import fnmatch
import os
import sys

COMPILE_FMT = "g++ {0} gtest/gmock-gtest-all.o gtest/gtest_main.o -I../ -lopencv_core"

def main():
    f = sys.argv[1]
    print "Running " + f
    os.system(COMPILE_FMT.format(f))
    os.system("./a.out")

main()
