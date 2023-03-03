// XFAIL: *
// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt
func.func @main(%arg0: !llvm.ptr<f64>, %arg1: i64) -> f64 {
  %2 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
  %3 = llvm.load %2 : !llvm.ptr<f64>
  %4 = llvm.load %2 : !llvm.ptr<f64>
  return %3 : f64
}
