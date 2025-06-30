add_mlir_dialect_library(MLIRNorthStarDialect3
  NorthStarDialect.cpp
  NorthStarTypes.cpp
  NorthStarAttrs.cpp
  NorthStarOps.cpp

  DEPENDS
  NorthStarDialectIncGen3

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorDialect
  )