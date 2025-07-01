#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::north_star {
std::unique_ptr<mlir::Pass> createApplyDistributeTransformPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Dialect/NorthStar/Transforms/Passes.h.inc"
}  // namespace north_star