#include "Dialect/NorthStar/NorthStarDialect.h"
#include "Dialect/NorthStar/NorthStarTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
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

  auto ui16 = mlir::IntegerType::get(context, 16, mlir::IntegerType::Unsigned);
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
  auto dynamic_affine_memref =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic, 2, 3}, f32,
                            mlir::StridedLayoutAttr::get(
                                context, 1, {mlir::ShapedType::kDynamic, 3, 1})
                                .getAffineMap());
  llvm::outs() << "连续附带 affine 布局信息的动态 F32 内存类型 :\t";
  dynamic_affine_memref.dump();

  // 具有内存层级信息的内存
  auto L1_memref =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic, 2, 3}, f32,
                            mlir::StridedLayoutAttr::get(
                                context, 1, {mlir::ShapedType::kDynamic, 3, 1})
                                .getAffineMap(),
                            1);
  llvm::outs() << "处于L1层级的 F32 内存类型 :\t";
  L1_memref.dump();

  // gpu 私有内存层级的内存, 电脑得有gpu
  //   context->getOrLoadDialect<mlir::gpu::GPUDialect>();
  //   auto gpu_memref =
  //       mlir::MemRefType::get({mlir::ShapedType::kDynamic, 2, 3}, f32,
  //                             mlir::StridedLayoutAttr::get(
  //                                 context, 1, {mlir::ShapedType::kDynamic, 3,
  //                                 1}) .getAffineMap(),
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
  auto dialect = context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
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
int main() { myType(); }
