#include <cstdint>
#include <memory>

#include "Dialect/NorthStar/NorthStarAttrs.h"
#include "Dialect/NorthStar/NorthStarDialect.h"
#include "Dialect/NorthStar/NorthStarEnums.h"
#include "Dialect/NorthStar/NorthStarOps.h"
#include "Dialect/NorthStar/NorthStarTypes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

void typeBrief() {
    auto context = new mlir::MLIRContext();

    // 浮点数, 位宽
    auto f32 = mlir::Float32Type::get(context);
    llvm::outs() << "f32类型: \t";
    f32.dump();

    auto bf16 = mlir::BFloat16Type::get(context);
    llvm::outs() << "bf16类型: \t";
    bf16.dump();

    auto index = mlir::IndexType::get(context);
    llvm::outs() << "index类型: \t";
    index.dump();

    auto i32 = mlir::IntegerType::get(context, 32);
    llvm::outs() << "i32类型: \t";
    i32.dump();

    auto ui16 =
        mlir::IntegerType::get(context, 16, mlir::IntegerType::Unsigned);
    llvm::outs() << "ui16类型: \t";
    ui16.dump();

    // 张量类型
    auto static_tensor = mlir::RankedTensorType::get({1, 2, 3}, f32);
    llvm::outs() << "静态F32 张量类型: \t";
    static_tensor.dump();

    auto dynamic_tensor =
        mlir::RankedTensorType::get({mlir::ShapedType::kDynamic, 2, 3}, f32);
    llvm::outs() << "动态F32 张量类型: \t";
    dynamic_tensor.dump();

    // Memref类型：表示内存
    auto basic_memref = mlir::MemRefType::get({1, 2, 3}, f32);
    llvm::outs() << "静态F32 内存类型 :\t";
    basic_memref.dump();

    // 带有布局信息的内存
    auto stride_layout_memref = mlir::MemRefType::get(
        {1, 2, 3}, f32, mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1}));
    llvm::outs() << "连续附带布局信息的 F32 内存类型: \t";
    stride_layout_memref.dump();

    // 使用affine 表示布局信息的内存
    auto affine_memref = mlir::MemRefType::get(
        {1, 2, 3}, f32,
        mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1}).getAffineMap());
    llvm::outs() << "affine 表示布局信息的 F32 内存类型: \t";
    affine_memref.dump();

    // 动态连续附带 affine 布局信息的内存
    auto dynamic_affine_memref = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, 2, 3}, f32,
        mlir::StridedLayoutAttr::get(context, 1,
                                     {mlir::ShapedType::kDynamic, 3, 1})
            .getAffineMap());
    llvm::outs() << "连续附带 affine 布局信息的动态 F32 内存类型 :\t";
    dynamic_affine_memref.dump();

    // 具有内存层级信息的内存
    auto L1_memref = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, 2, 3}, f32,
        mlir::StridedLayoutAttr::get(context, 1,
                                     {mlir::ShapedType::kDynamic, 3, 1})
            .getAffineMap(),
        1);
    llvm::outs() << "处于L1层级的 F32 内存类型 :\t";
    L1_memref.dump();

    // gpu 私有内存层级的内存, 电脑得有gpu
    //   context->getOrLoadDialect<mlir::gpu::GPUDialect>();
    //   auto gpu_memref =
    //       mlir::MemRefType::get({mlir::ShapedType::kDynamic, 2, 3}, f32,
    //                             mlir::StridedLayoutAttr::get(
    //                                 context, 1, {mlir::ShapedType::kDynamic,
    //                                 3, 1}) .getAffineMap(),
    //                             mlir::gpu::AddressSpaceAttr::get(
    //                                 context,
    //                                 mlir::gpu::AddressSpace::Private));
    //   llvm::outs() << "连续附带 affine 布局信息的动态 F32 Gpu Private内存类型
    //   :\t"; gpu_memref.dump();

    // 向量类型，定长一段内存
    auto vector_type = mlir::VectorType::get(3, f32);
    llvm::outs() << "定长1D 3 个 F32 向量类型 :\t";
    vector_type.dump();

    auto vector_2D_type = mlir::VectorType::get({3, 3}, f32);
    llvm::outs() << "定长2D 3x3 个 F32 向量类型 :\t";
    vector_2D_type.dump();
    delete context;
}

void myType() {
    typeBrief();
    // 初始化方言注册器
    mlir::DialectRegistry registry;
    // 初始化上下文环境
    mlir::MLIRContext context(registry);
    // 加载/注册方言
    auto dialect =
        context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
    dialect->sayHello();
    // 静态 NSTensor
    auto ns_tensor = mlir::north_star::NSTensorType::get(
        &context, {1, 2, 3}, mlir::Float32Type::get(&context), 3);
    ns_tensor.dump();

    auto dy_ns_tensor = mlir::north_star::NSTensorType::get(
        &context, {mlir::ShapedType::kDynamic, 2, 3},
        mlir::Float32Type::get(&context));
    llvm::outs() << "Dynamic NSTensor: " << dy_ns_tensor << "\n";
    dy_ns_tensor.dump();
}

void attributeBrief() {
    //   auto context = new mlir::MLIRContext;
    auto context = std::make_shared<mlir::MLIRContext>();
    context->getOrLoadDialect<mlir::north_star::NorthStarDialect>();

    auto f32_attr =
        mlir::FloatAttr::get(mlir::Float32Type::get(context.get()), 2);
    llvm::outs() << "F32 Attribute: \t";
    f32_attr.dump();

    auto i32_attr =
        mlir::IntegerAttr::get(mlir::IntegerType::get(context.get(), 32), 10);
    llvm::outs() << "I32 Attribute: \t";
    i32_attr.dump();

    auto stride_layout_attr =
        mlir::StridedLayoutAttr::get(context.get(), 1, {6, 3, 1});
    llvm::outs() << "Stride Layout Attribute: \t";
    stride_layout_attr.dump();

    auto str_attr = mlir::StringAttr::get(context.get(), "Hello MLIR");
    llvm::outs() << "String Attribute: \t";
    str_attr.dump();

    auto str_ref_attr = mlir::SymbolRefAttr::get(str_attr);
    llvm::outs() << "Symbol Ref Attribute: \t";
    str_ref_attr.dump();

    auto type_attr = mlir::TypeAttr::get(mlir::north_star::NSTensorType::get(
        context.get(), {1, 2, 3}, mlir::Float32Type::get(context.get())));
    llvm::outs() << "Type Attribute: \t";
    type_attr.dump();

    auto unit_attr = mlir::UnitAttr::get(context.get());
    llvm::outs() << "Unit Attribute: \t";
    unit_attr.dump();

    auto i64_arr_attr = mlir::DenseI64ArrayAttr::get(context.get(), {1, 2, 3});
    llvm::outs() << "Dense I64 Array Attribute: \t";
    i64_arr_attr.dump();

    auto dense_attr = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get({2, 2},
                                    mlir::Float32Type::get(context.get())),
        llvm::ArrayRef<float>{1, 2, 3, 4});
    llvm::outs() << "Dense Elements Attribute: \t";
    dense_attr.dump();
}

void myAttr() {
    attributeBrief();
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    auto dialect =
        context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
    auto nchw = mlir::north_star::Layout::NCHW;
    llvm::outs() << "Layout: " << mlir::north_star::stringifyEnum(nchw) << "\n";

    auto nchw_attr = mlir::north_star::LayoutAttr::get(&context, nchw);
    llvm::outs() << "NCHW LayoutAttribute: \t";
    nchw_attr.dump();

    auto dp_attr = mlir::north_star::DataParallelismAttr::get(&context, 2);
    llvm::outs() << "DataParallelism Attribute: \t";
    dp_attr.dump();
    delete dialect;
}

void myOperation() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = builder.create<mlir::ModuleOp>(loc, "NorthStar");
    builder.setInsertionPointToEnd(module.getBody());
    auto f32 = mlir::Float32Type::get(&context);
    auto shape = mlir::SmallVector<int64_t>{2, 2};
    auto const_value1 =
        mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)1));
    auto const_value2 =
        mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)2));
    auto tensor_type1 =
        mlir::north_star::NSTensorType::get(&context, shape, f32, 0);
    auto tensor_type2 =
        mlir::north_star::NSTensorType::get(&context, shape, f32, 1);
    auto const1 = builder.create<mlir::north_star::ConstOp>(
        loc, tensor_type1,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value1));
    auto const2 = builder.create<mlir::north_star::ConstOp>(
        loc, tensor_type1,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value1));
    auto const3 = builder.create<mlir::north_star::ConstOp>(
        loc, tensor_type2,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value2));
    auto const4 = builder.create<mlir::north_star::ConstOp>(
        loc, tensor_type2,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value2));
    llvm::outs() << "Const tensor in device 0: \n";
    const1->dump();
    llvm::outs() << "Const tensor in device 1: \n";
    const3->dump();

    // buffer of
    auto buffer_op = builder.create<mlir::north_star::BufferOp>(
        loc, mlir::ValueRange({const1, const3}));
    llvm::outs() << "Buffer Op: \n";
    buffer_op->dump();

    // Get tensor op
    auto get_tensor_op1 = builder.create<mlir::north_star::GetTensorOp>(
        loc, tensor_type1, buffer_op, 0);
    auto get_tensor_op2 = builder.create<mlir::north_star::GetTensorOp>(
        loc, tensor_type2, buffer_op, 1);
    llvm::outs() << "Get tensor op: \n";
    get_tensor_op1->dump();
    get_tensor_op2->dump();

    // softmax op
    auto softmax_op =
        builder.create<mlir::north_star::SoftmaxOp>(loc, get_tensor_op1, 1);
    llvm::outs() << "Softmax op: \n";
    softmax_op->dump();

    // exp op
    auto exp_op = builder.create<mlir::north_star::ExpOp>(loc, get_tensor_op2);
    llvm::outs() << "Exp op: \n";
    exp_op->dump();

    // all to all op
    auto out_buffer_op = builder.create<mlir::north_star::BufferOp>(
        loc, mlir::ValueRange({const2, const4}));
    auto all_to_all_op = builder.create<mlir::north_star::AllToAllOp>(
        loc, buffer_op, out_buffer_op);
    llvm::outs() << "All to All Op :\n";
    all_to_all_op->dump();
}

void myInterface() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
    // context.getOrLoadDialect<mlir::func::FuncDialect>();

    auto f32 = mlir::Float32Type::get(&context);
    auto dim = mlir::ShapedType::kDynamic;
    auto shape = mlir::SmallVector<int64_t>{dim, dim, 24};
    auto tensor_type =
        mlir::north_star::NSTensorType::get(&context, shape, f32, 0);
    auto shaped_type = mlir::cast_or_null<mlir::ShapedType>(tensor_type);
    llvm::outs() << "NSTensorType: \t";
    shaped_type.dump();
    llvm::outs() << "Shaped Type Interface: \t";
    shaped_type.dump();

    auto cloned_type = shaped_type.clone(f32);
    llvm::outs() << "Cloned Shaped Type Interface:\t";
    cloned_type.dump();

    auto dp_attr = mlir::north_star::DataParallelismAttr::get(&context, 2);
    llvm::outs()
        << dp_attr.getAbstractAttribute().getName()
        << " has DistributeParallelAttr: "
        << dp_attr
               .hasPromiseOrImplementsInterface<mlir::DistributeParallelAttr>()
        << "\n";
}

int main() {
    myInterface();
    return 0;
}
