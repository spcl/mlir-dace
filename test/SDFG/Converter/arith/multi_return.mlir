// XFAIL: *
// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt
func.func private @main(%arg1: i32, %arg2: i32) -> (i32, i32) {
  %low, %high = arith.mulsi_extended %arg1, %arg2 : i32
  return %low, %high : i32, i32
}
