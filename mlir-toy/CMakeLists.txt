cmake_minimum_required(VERSION 3.20)
project(mlir-toy LANGUAGES C CXX)

cmake_policy(SET CMP0116 NEW)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_INSTALL_PREFIX "/home/cyj/workspace/mlir-noob/install")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(ENV{PATH} "/home/cyj/workspace/mlir-noob/install/bin:\$ENV{PATH}")

add_compile_options(-g)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

add_executable(mlir-toy main.cpp)

add_executable(ex1-io ex1-io/ex1-io.cpp)

target_link_libraries(ex1-io PRIVATE 
    MLIRIR 
    MLIRParser 
    MLIRFuncDialect
    MLIRArithDialect
)

add_subdirectory(dialect)
add_subdirectory(north_star_dialect)
add_subdirectory(north_star_type)
add_subdirectory(north_star_attribute)
add_subdirectory(north_star_operation)
add_subdirectory(north_star_interface)
add_subdirectory(north_star_pass)