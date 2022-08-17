// RUN: sdfg-opt --convert-to-sdfg %s
func.func private @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg11 = %c0 to %c1 step %c1 {
    %v2 = arith.addi %arg11, %arg11 : index
  }
  return
}
