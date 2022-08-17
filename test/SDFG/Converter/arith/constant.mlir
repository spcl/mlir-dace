// RUN: sdfg-opt --convert-to-sdfg %s
func.func private @main() {
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f64
  return
}
