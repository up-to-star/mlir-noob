include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include/Dialect/NorthStar/IR)
add_subdirectory(include/Interfaces)
add_subdirectory(include/Dialect/NorthStar/Transforms)
add_subdirectory(src/Interfaces)
add_subdirectory(src/Dialect/NorthStar/IR)
add_subdirectory(src/Dialect/NorthStar/Transforms)

add_executable(north_star_pass_test main.cpp)
target_link_libraries(north_star_pass_test MLIRNorthStarDialect5 MLIRFuncDialect MLIRNorthStarTransforms1)
