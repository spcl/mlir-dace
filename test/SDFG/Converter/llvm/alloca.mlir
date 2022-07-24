// RUN: sdfg-opt --convert-to-sdfg %s
func.func @name(){
  %c1_i64 = arith.constant 1 : i64
  %1 = llvm.alloca %c1_i64 x !llvm.ptr<f64> : (i64) -> !llvm.ptr<ptr<f64>>
  return
}
