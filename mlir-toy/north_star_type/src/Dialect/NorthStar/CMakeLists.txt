add_mlir_dialect_library(MLIRNorthStarDialect1
  NorthStarDialect.cpp
  NorthStarTypes.cpp

  DEPENDS
  NorthStarDialectIncGen1

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorDialect
  )