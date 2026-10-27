cmake_minimum_required(VERSION 3.16)
project(flycore_lif LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(flycore_lif STATIC lif_controller.cpp)
target_include_directories(flycore_lif PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Shared library consumed from Python via ctypes (flycore/cpp_bridge.py).
add_library(flycore_lif_shared SHARED lif_controller.cpp)
set_target_properties(flycore_lif_shared PROPERTIES OUTPUT_NAME flycore_lif)
target_include_directories(flycore_lif_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
