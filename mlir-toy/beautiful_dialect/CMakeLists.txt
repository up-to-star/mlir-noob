set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(include/toy)