#include "Dialect/NorthStar/IR/NorthStarAttrs.h"
#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarEnums.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/NorthStar/IR/NorthStarAttrs.cpp.inc"
#include "Dialect/NorthStar/IR/NorthStarEnums.cpp.inc"

namespace mlir::north_star {
    void NorthStarDialect::registerAttrs() {
        llvm::outs() << "Registering NorthStarDialect attributes..." << getDialectNamespace() << "\n";
        addAttributes<
        #define GET_ATTRDEF_LIST
        #include "Dialect/NorthStar/IR/NorthStarAttrs.cpp.inc"
        >();
    }

    bool LayoutAttr::isChannelLast() {
        return getValue() == Layout::NHWC;
    }

}

