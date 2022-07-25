// RUN: sdfg-opt --convert-to-sdfg %s
func.func private @kernel_2mm() {
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f64
  return
}
