add_mlir_dialect_library(MLIRNorthStarDialect
  NorthStarDialect.cpp

  DEPENDS
  NorthStarDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorDialect
  )