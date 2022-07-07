// RUN: sdfg-opt --convert-to-sdfg %s
func.func private @ex(i32, i32) -> (i32)

func.func private @kernel(%arg1: i32, %arg2: i32) {
  %c0 = arith.addi %arg1, %arg2 : i32
  %res = func.call @ex(%c0, %c0) : (i32, i32) -> (i32)
  %c1 = arith.addi %res, %res : i32
  return
}
