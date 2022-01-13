// RUN: sdir-opt --convert-to-sdir %s
func private @kernel_2mm(%arg1: i32, %arg2: i32) {
  %c0 = arith.addi %arg1, %arg2 : i32
  %c2 = arith.muli %arg1, %arg2 : i32
  return
}
