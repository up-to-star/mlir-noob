add_mlir_dialect_library(MLIRNorthStarDialect4
  NorthStarDialect.cpp
  NorthStarTypes.cpp
  NorthStarAttrs.cpp
  NorthStarOps.cpp

  DEPENDS
  NorthStarDialectIncGen4

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorDialect
  )