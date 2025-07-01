set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name NorthStarOpt)
add_public_tablegen_target(MLIRNorthStarPassesIncGen1)
add_dependencies(mlir-headers MLIRNorthStarPassesIncGen1)