include "mlir/Pass/PassBase.td"


def MarkDistributeParallelParametersPass : Pass<"mark-distribute-parallel-parameters", "::mlir::ModuleOp"> {
    let summary = "Mark global distribute parallel parameter";
    let description = [{
        "标记全局并行参数";
    }];

    let dependentDialects = [
        "::mlir::north_star::NorthStarDialect",
        "::mlir::tensor::TensorDialect"
    ];
    let options = [
        Option<"DPNums", "DP", "std::int64_t", /*default=*/"1", "DPNums des">,
        Option<"TPNums", "TP", "std::int64_t", /*defalut=*/"1", "TPNums des">
    ];

    let statistics = [
        Statistic<"EPNums", "ep-nums", "Number of EP">
    ];
}

def ApplyDistributeTransformPass : Pass<"apply-distribute-transform", "::mlir::func::FuncOp"> {
    let summary = "根据标记的并行参数进行变换";
    let description = [{
        "根据标记的并行参数进行变换。"
    }];

    let dependentDialects = [
        "::mlir::north_star::NorthStarDialect",
        "::mlir::tensor::TensorDialect"
    ];
    let constructor = "mlir::north_star::createApplyDistributeTransformPass()";
}