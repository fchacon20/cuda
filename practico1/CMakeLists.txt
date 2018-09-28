cmake_minimum_required(VERSION 3.7)
project(cuda1)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp functions.cpp functions.h)
add_executable(cuda1 ${SOURCE_FILES} functions.h)