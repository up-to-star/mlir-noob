#include "Dialect/NorthStar/NorthStarDialect.h"
#include "include/Dialect/NorthStar/NorthStarDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

int main() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context;
    auto dialect = context.getLoadedDialect<mlir::north_star::NorthStarDialect>();
    dialect->sayHello();
}
