cmake_minimum_required(VERSION 3.20)
project(DLSE LANGUAGES CXX)

option(DLSE_ENABLE_CUDA "Build DLSE CUDA kernels and benchmarks" OFF)
option(DLSE_ENABLE_SANITIZERS "Use AddressSanitizer and UndefinedBehaviorSanitizer for host targets" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(DLSE_ENABLE_SANITIZERS AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "DLSE_ENABLE_SANITIZERS requires GCC or Clang")
endif()

function(dlse_apply_sanitizers target)
    if(DLSE_ENABLE_SANITIZERS)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

enable_testing()

add_library(dlse_runtime_core src/runtime.cpp)
target_include_directories(dlse_runtime_core PUBLIC include)
target_compile_options(dlse_runtime_core PRIVATE -Wall -Wextra -Wpedantic)
dlse_apply_sanitizers(dlse_runtime_core)

add_executable(dlse_runtime_tests tests/runtime_tests.cpp)
target_link_libraries(dlse_runtime_tests PRIVATE dlse_runtime_core)
target_compile_options(dlse_runtime_tests PRIVATE -Wall -Wextra -Wpedantic)
dlse_apply_sanitizers(dlse_runtime_tests)

add_test(NAME dlse_scheduler_tests COMMAND dlse_runtime_tests scheduler)
add_test(NAME dlse_kv_cache_tests COMMAND dlse_runtime_tests kv)
add_test(NAME dlse_stress_tests COMMAND dlse_runtime_tests stress)
set_tests_properties(
    dlse_scheduler_tests
    dlse_kv_cache_tests
    dlse_stress_tests
    PROPERTIES TIMEOUT 30
)

add_executable(dlse_runtime_bench benchmarks/runtime_bench.cpp)
target_link_libraries(dlse_runtime_bench PRIVATE dlse_runtime_core)
target_compile_options(dlse_runtime_bench PRIVATE -Wall -Wextra -Wpedantic)
dlse_apply_sanitizers(dlse_runtime_bench)

add_executable(dlse_runtime_experiments benchmarks/runtime_experiments.cpp)
target_link_libraries(dlse_runtime_experiments PRIVATE dlse_runtime_core)
target_compile_options(dlse_runtime_experiments PRIVATE -Wall -Wextra -Wpedantic)
dlse_apply_sanitizers(dlse_runtime_experiments)

if(DLSE_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures for DLSE")
    endif()

    add_library(dlse_cuda STATIC cuda/paged_attention.cu)
    target_include_directories(dlse_cuda PUBLIC include)
    target_link_libraries(dlse_cuda PUBLIC CUDA::cudart)
    set_target_properties(dlse_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

    add_executable(dlse_paged_attention_bench benchmarks/paged_attention_bench.cu)
    target_link_libraries(dlse_paged_attention_bench PRIVATE dlse_cuda CUDA::cudart)
    set_target_properties(dlse_paged_attention_bench PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

    add_executable(dlse_cuda_graph_bench benchmarks/cuda_graph_bench.cu)
    target_link_libraries(dlse_cuda_graph_bench PRIVATE CUDA::cudart)
    set_target_properties(dlse_cuda_graph_bench PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()
