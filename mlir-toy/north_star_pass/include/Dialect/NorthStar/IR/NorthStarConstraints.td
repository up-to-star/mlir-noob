include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"


def AnyNSTensor : Type<And<[CPred<"::mlir::isa<::mlir::north_star::NSTensorType>($_self)">,
                            CPred<"::mlir::cast<::mlir::north_star::NSTensorType>($_self).getShape().size() > 0">]>>;


def AnyBuffer : Type<And<[CPred<"::mlir::isa<::mlir::north_star::BufferType>($_self)">,
                          CPred<"::mlir::cast<::mlir::north_star::BufferType>($_self).getDevices().size() > 0">]>>;

def NSTensorOrBuffer : AnyTypeOf<[AnyNSTensor, AnyBuffer]>;