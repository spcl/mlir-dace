// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt
func.func private @main(%arg0: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg11 = %c0 to %c1 step %c1 {
    %v2 = arith.addi %arg0, %arg0 : i32
  }
  return
}
