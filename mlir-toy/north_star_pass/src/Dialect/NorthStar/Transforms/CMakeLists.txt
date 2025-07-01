add_mlir_dialect_library(MLIRNorthStarTransforms1
  ApplyDistributeTransform.cpp
  MarkDistributeParallelParameters.cpp

  DEPENDS
  MLIRNorthStarPassesIncGen1

  LINK_LIBS PUBLIC
  MLIRNorthStarDialect5
)
