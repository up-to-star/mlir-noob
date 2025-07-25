include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "Dialect/NorthStar/NorthStarDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class NorthStar_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<NorthStar_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName =  dialect.name # "." # typeMnemonic;
}

def NorthStar_BufferType : NorthStar_Type<"Buffer", "buffer", []> {
  let summary = "the summary of buffer type";
  let description = "description of buffer type";
  let parameters = (ins
    ArrayRefParameter<"int64_t ">: $devices
  );
  let genStorageClass = 1;
  let hasStorageCustomConstructor = 0;
  let assemblyFormat = "`<`$devices`>`";
  let skipDefaultBuilders = 0;
  let genVerifyDecl = 1;
}


def NorthStar_TensorType : NorthStar_Type<"NSTensor","ns_tensor",[ShapedTypeInterface], "::mlir::TensorType">{
  // 概述
  let summary = " the summary of north-star tensor type";

  // 方言的详细描述
  let description = "description of north-star tensor type";

  // 参数
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "int64_t":$device_id
  );

  // 是否生成StorageClass, 无特殊情况，建议设为ture
  let genStorageClass = 1;
  
  // 不建议改动
  let hasStorageCustomConstructor = 0;

  // 额外的builder 声明
  let builders = [
    TypeBuilder<(ins 
        "::mlir::ArrayRef<int64_t>":$shape,
        "::mlir::Type":$elementType),[{
      return $_get(elementType.getContext(), shape, elementType, 0);
    }]>
  ];

//   let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "`<`$shape`,`$elementType`,`$device_id`>`";

  // 跳过默认的builder函数
  let skipDefaultBuilders = 0;

  // 是否生成类型检验的函数声明
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
     using ::mlir::ShapedType::Trait<NSTensorType>::getElementTypeBitWidth;
    using ::mlir::ShapedType::Trait<NSTensorType>::getRank;
    using ::mlir::ShapedType::Trait<NSTensorType>::getNumElements;
    using ::mlir::ShapedType::Trait<NSTensorType>::isDynamicDim;
    using ::mlir::ShapedType::Trait<NSTensorType>::hasStaticShape;
    using ::mlir::ShapedType::Trait<NSTensorType>::getNumDynamicDims;
    using ::mlir::ShapedType::Trait<NSTensorType>::getDimSize;
    using ::mlir::ShapedType::Trait<NSTensorType>::getDynamicDimIndex;

    ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type type){
        if(shape)
          return mlir::north_star::NSTensorType::get(getContext(), *shape, type);
        return mlir::north_star::NSTensorType::get(getContext(), getShape(), type);
    }
    NSTensorType clone(::mlir::ArrayRef<int64_t> shape,int64_t device_id) const {
      return NSTensorType::get(getContext(), shape, getElementType(),device_id);
    }
    NSTensorType clone(::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType) const {
      return NSTensorType::get(getContext(), shape, elementType,getDeviceId());
    }
    NSTensorType clone(::mlir::ArrayRef<int64_t> shape) const {
      return NSTensorType::get(getContext(), shape, getElementType(),getDeviceId());
    }
    NSTensorType clone(::mlir::Type elementType) const {
      return NSTensorType::get(getContext(), getShape(), elementType,getDeviceId());
    }
    NSTensorType clone() const {
      return NSTensorType::get(getContext(), getShape(), getElementType(),getDeviceId());
    }
  }];
}