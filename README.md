LinAlg
======

Linear Algebra functions

Dependencies
------------
* BLAS (openblas but can be changed)
* LAPACK
* GoogleTest for testing
* CMake and make for building

Building
--------
To build the project make a directory called build, change in to it and
then run `cmake ..`. After this typing make will build the library.

```
mkdir build
cd build
cmake ..
make
```

To run the unit tests more work is required.  Firstly, you will need to
have Google Test installed.  Create a link to google test then follow
the above commands passing -Dbvls_build_tests=ON.

```
ln -s ${YOUR_GTEST_INSTALL_DIR} gtest
mkdir build
cd build
cmake -Dbvls_build_tests=ON
make
make test
```

Alternatively the test can be run by issuing the command `bvls_test`
after building.
