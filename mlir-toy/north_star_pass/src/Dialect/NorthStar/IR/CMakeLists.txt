add_mlir_dialect_library(MLIRNorthStarDialect5
  NorthStarDialect.cpp
  NorthStarTypes.cpp
  NorthStarAttrs.cpp
  NorthStarOps.cpp

  DEPENDS
  NorthStarDialectIncGen5

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorDialect
  MLIRDistributeParallelismInterfaces1
  )