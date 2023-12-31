
cmake_minimum_required(VERSION 3.28)

project(MLFramework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(MLFramework src/main.cpp)

target_compile_options(MLFramework PUBLIC 
    -std=c++17 
    -Werror 
    -Weverything
    -Wno-c++98-compat
    -Wno-c++98-c++11-c++14-compat
    -Wno-c++98-compat-pedantic)

target_include_directories(MLFramework PRIVATE include)

add_subdirectory(test)
