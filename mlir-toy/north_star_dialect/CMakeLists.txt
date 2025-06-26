include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include/Dialect/NorthStar)
add_subdirectory(src/Dialect/NorthStar)

add_executable(north_star_dialect_test main.cpp)
target_link_libraries(north_star_dialect_test MLIRNorthStarDialect)
