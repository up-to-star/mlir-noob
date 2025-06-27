add_mlir_dialect_library(MLIRNorthStarDialect2
  NorthStarDialect.cpp
  NorthStarTypes.cpp
  NorthStarAttrs.cpp

  DEPENDS
  NorthStarDialectIncGen2

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorDialect
  )