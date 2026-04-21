cmake_minimum_required(VERSION 3.10)
project(tiny_cls_ncnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable(tiny_cls_infer infer.cpp)
target_link_libraries(tiny_cls_infer PRIVATE ncnn ${OpenCV_LIBS})

