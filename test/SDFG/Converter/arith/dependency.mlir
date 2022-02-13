// RUN: sdfg-opt --convert-to-sdfg %s
func private @kernel_2mm(%arg1: i32) {
  %c0 = arith.addi %arg1, %arg1 : i32
  %c2 = arith.muli %c0, %c0 : i32
  return
}
