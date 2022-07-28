// RUN: sdfg-opt --convert-to-sdfg %s
func.func @main(%arg0: !llvm.ptr<i8>) -> !llvm.ptr<f64>{
  %1 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<f64>
  return %1 : !llvm.ptr<f64>
}
