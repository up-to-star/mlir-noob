#pragma once
#include "Dialect/NorthStar/IR/NorthStarEnums.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"

#define FIX
#include "Interfaces/DistributeParallelismAttrInterfaces.h.inc"
#include "Interfaces/DistributeParallelismOpInterfaces.h.inc"
#undef FIX