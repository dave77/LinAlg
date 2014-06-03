project(bvls)

cmake_minimum_required(VERSION 2.8)

option(bvls_build_tests "Build the unit tests for the BVLS solver." OFF)

set(CMAKE_C_FLAGS "-g -O2 -std=c99 -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS "-g -O2 -Wall -Wextra")

file(GLOB bvls_HEADERS include/*)
file(GLOB bvls_SOURCES src/*.c)

include_directories(include)
add_library(bvls STATIC ${bvls_SOURCES})
add_library(bvls_shared SHARED ${bvls_SOURCES})
set_target_properties(bvls_shared PROPERTIES COMPILE_FLAGS "-fPIC")

if (bvls_build_tests)
	enable_testing()
	
	add_subdirectory(gtest)
	file(GLOB test_bvls_SOURCES test/*.cpp)
	include_directories(gtest/include)
	add_executable(test_bvls ${test_bvls_SOURCES})
	target_link_libraries(test_bvls bvls gtest_main gtest -lopenblas -lpthread)
	add_test(BVLS test_bvls)
endif()
