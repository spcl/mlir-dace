// RUN: sdfg-opt --convert-to-sdfg %s
func private @kernel_2mm(%arg1: i32, %arg2: i32, %arg3: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg2 : i32 to index
  %2 = arith.index_cast %arg3 : i32 to index
  return
}
