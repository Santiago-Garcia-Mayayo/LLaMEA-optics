# CEC 2020 - Real World Constrained Benchmark Functions in C++ (and Python)

The C++ implementation of the test suite for the Congress on Evolutionary Computation 2020 Competition on Real-World Constrained Optimization.

Compilation performed with GCC 9.4.0, tested on Ubuntu Linux 20.04 and Windows 10.

# Dependencies

The only dependence is the [Eigen](https://gitlab.com/libeigen/eigen) library, so make sure that it is present in the folder where you compile the code. Eigen version 3.4.0 was used during development.

# C++ instructions

The test.cpp file contains a usage example. To compile it run:

g++ -std=c++14 test.cpp -o test.out

or add some optimizations:

g++ -std=c++14 -march=corei7-avx -fexpensive-optimizations -O3 test.cpp -o test.out

# Python wrapper instructions

A very simple python wrapper is included here. To use it, first you need to compile the file with the functions:

g++ -std=c++14 -fPIC -shared -lm -o cec2020rwc.so cec2020rwc.cpp

or add some optimizations:

g++ -std=c++14 -march=corei7-avx -fexpensive-optimizations -O3 -fPIC -shared -lm -o cec2020rwc.so cec2020rwc.cpp

This will create a file cec2020rwc.so which is going to be used by the cec2020rwc.py, where two functions are defined, get_bounds and cec2020rwc. There is no pip installation, so you would need to manually include these files. The example is shown in test.py.

# Original CEC 2020 code

The original code is written in Matlab by Abhishek Kumar, it can be found at https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation

# Known differences from Matlab

RC24: in C++ acos is calculated differently from matlab, resulting in small differences (<1e-10 relative);

RC31: should have 0 g's and 0 h's, but it has 1's - probably to avoid empty arrays? Also, it is just a box-constrained problem;

RC44: g and h should be 91 not 105.
