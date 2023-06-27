// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt
func.func private @main() -> (i32, i32) {
  %init1 = arith.constant 0 : i32
  %init2 = arith.constant 1 : i32

  %res:2 = scf.while (%arg1 = %init1, %arg2 = %init2) : (i32, i32) -> (i32, i32) {
    %c10 = arith.constant 10 : i32
    %condition = arith.cmpi sle, %arg1, %c10 : i32
    scf.condition(%condition) %arg1, %arg2 : i32, i32
  } do {
  ^bb0(%arg3: i32, %arg4: i32):
    %sum = arith.addi %arg3, %arg4 : i32
    scf.yield %sum, %sum : i32, i32
  }
  
  return %res#0, %res#1 : i32, i32
}
