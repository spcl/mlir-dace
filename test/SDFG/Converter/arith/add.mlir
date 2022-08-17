// RUN: sdfg-opt --convert-to-sdfg %s
func.func private @main(%arg1: i32, %arg2: i32) -> i32 {
  %c0 = arith.addi %arg1, %arg2 : i32
  return %c0 : i32
}
