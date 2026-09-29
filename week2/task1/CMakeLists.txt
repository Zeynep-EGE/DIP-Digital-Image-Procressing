cmake_minimum_required(VERSION 3.20)

set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

project(task1)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
add_executable(task1 main.cpp)
target_link_libraries(task1 ${OpenCV_LIBS})
