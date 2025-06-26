set(LLVM_TARGET_DEFINITIONS NorthStarDialect.td)

mlir_tablegen(NorthStarDialect.h.inc --gen-dialect-decls -dialect=north_star)
mlir_tablegen(NorthStarDialect.cpp.inc --gen-dialect-defs -dialect=north_star)

add_public_tablegen_target(NorthStarDialectIncGen)
