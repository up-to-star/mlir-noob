add_mlir_library(MLIRDistributeParallelismInterfaces1
    DistributeParallelismInterfaces.cpp

    DEPENDS
    MLIRDistributeParallelismInterfacesIncGen1

    LINK_LIBS PUBLIC
    MLIRIR
)