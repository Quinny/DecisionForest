import fnmatch
import os

COMPILE_FMT = "g++ {0} gtest/gmock-gtest-all.o gtest/gtest_main.o -I../"

def all_test_files():
    return filter(lambda f: fnmatch.fnmatch(f, "*_test.cpp"), os.listdir("."))

def main():
    for f in all_test_files():
        print "Running " + f
        os.system(COMPILE_FMT.format(f))
        os.system("./a.out")

main()
