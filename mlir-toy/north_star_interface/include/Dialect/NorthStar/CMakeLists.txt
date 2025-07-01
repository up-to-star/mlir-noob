set(LLVM_TARGET_DEFINITIONS NorthStarTypes.td)

mlir_tablegen(NorthStarDialect.h.inc --gen-dialect-decls -dialect=north_star)
mlir_tablegen(NorthStarDialect.cpp.inc --gen-dialect-defs -dialect=north_star)

mlir_tablegen(NorthStarTypes.h.inc -gen-typedef-decls -dialect=north_star)
mlir_tablegen(NorthStarTypes.cpp.inc -gen-typedef-defs -dialect=north_star)

set(LLVM_TARGET_DEFINITIONS NorthStarEnums.td)
mlir_tablegen(NorthStarEnums.h.inc -gen-enum-decls -dialect=north_star)
mlir_tablegen(NorthStarEnums.cpp.inc -gen-enum-defs -dialect=north_star)

set(LLVM_TARGET_DEFINITIONS NorthStarAttrs.td)
mlir_tablegen(NorthStarAttrs.h.inc -gen-attrdef-decls -dialect=north_star)
mlir_tablegen(NorthStarAttrs.cpp.inc -gen-attrdef-defs -dialect=north_star)

set(LLVM_TARGET_DEFINITIONS NorthStarOps.td)
mlir_tablegen(NorthStarOps.h.inc -gen-op-decls -dialect=north_star)
mlir_tablegen(NorthStarOps.cpp.inc -gen-op-defs -dialect=north_star)

add_public_tablegen_target(NorthStarDialectIncGen4)
