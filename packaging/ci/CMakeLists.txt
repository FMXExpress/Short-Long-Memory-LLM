# CI diagnostic only: builds llama_load.c against an existing llama.cpp build
cmake_minimum_required(VERSION 3.16)
project(llama_load C)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
add_executable(llama_load llama_load.c)
target_include_directories(llama_load PRIVATE ${LLAMA_SRC}/include ${LLAMA_SRC}/ggml/include)
target_link_directories(llama_load PRIVATE ${LLAMA_SRC}/build/src/Release ${LLAMA_SRC}/build/ggml/src/Release)
target_link_libraries(llama_load PRIVATE llama ggml ggml-base)
