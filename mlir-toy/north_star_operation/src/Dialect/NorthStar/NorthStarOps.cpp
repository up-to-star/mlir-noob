#include "Dialect/NorthStar/NorthStarOps.h"

#include <algorithm>
#include <cstdint>

#include "Dialect/NorthStar/NorthStarDialect.h"
#include "Dialect/NorthStar/NorthStarTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#define GET_OP_CLASSES
#include "Dialect/NorthStar/NorthStarOps.cpp.inc"

namespace mlir::north_star {
void NorthStarDialect::registerOps() {
    llvm::outs() << "registering ops..." << getDialectNamespace() << " Op\n";
    addOperations<
#define GET_OP_LIST
#include "Dialect/NorthStar/NorthStarOps.cpp.inc"
        >();
}

::llvm::LogicalResult GetTensorOp::verify() {
    auto device_id = getDeviceId();
    auto buffer = getBuffer();
    if (llvm::isa<BlockArgument>(buffer)) {
        auto buffer_type = cast<BufferType>(buffer.getType());
        auto device_ids = buffer_type.getDevices();
        for (auto id : device_ids) {
            if (static_cast<uint64_t>(id) == device_id) {
                return mlir::success();
            }
        }
        return llvm::failure();
    }
    auto buffer_op = llvm::cast_or_null<BufferOp>(buffer.getDefiningOp());
    if (!buffer_op) return llvm::failure();
    for (auto tensor : buffer_op.getTensors()) {
        auto tensor_type = llvm::cast_or_null<NSTensorType>(tensor.getType());
        if (!tensor_type) return llvm::failure();
        if (static_cast<uint64_t>(tensor_type.getDeviceId()) == device_id) {
            if (tensor_type != getType()) return llvm::failure();
            return mlir::success();
        }
    }
    return llvm::failure();
}

::llvm::LogicalResult BufferOp::verify() {
    auto tensors = getTensors();
    auto devices = cast<BufferType>(getType()).getDevices();
    if (tensors.size() == 0) return llvm::failure();
    for (auto [idx, device_id, tensor] : llvm::enumerate(devices, tensors)) {
        auto tensor_type = llvm::cast_or_null<NSTensorType>(tensor.getType());
        if (device_id != tensor_type.getDeviceId()) return llvm::failure();
    }
    return llvm::success();
}

::llvm::LogicalResult SoftmaxOp::verify() {
    auto axis = getAxis();
    if (axis < 0) return llvm::failure();
    auto input_type = cast<NSTensorType>(getInput().getType());
    if (axis >= input_type.getShape().size()) return llvm::failure();
    return llvm::success();
}

}  // namespace mlir::north_star