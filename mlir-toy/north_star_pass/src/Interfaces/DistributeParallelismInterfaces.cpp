#include "Interfaces/DistributeParallelismInterfaces.h"

#include <cstdint>

#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "Interfaces/DistributeParallelismAttrInterfaces.cpp.inc"
#include "Interfaces/DistributeParallelismOpInterfaces.cpp.inc"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
void test() {
    int DP_nums;
    llvm::SmallVector<int64_t> device_ids;
    for (auto i : llvm::index_range(0, DP_nums)) {
        device_ids.push_back(i);
    }
}