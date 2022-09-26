// RUN: sdfg-opt --convert-to-sdfg %s
func.func private @main() {
  %init1 = arith.constant 0 : i32

  %res = scf.while (%arg1 = %init1, %arg2 = %init1) : (i32, i32) -> i32 {
    %shared = arith.constant 0 : i32
    %condition = arith.constant true
    %sum = arith.addi %arg1, %arg2 : i32
    scf.condition(%condition) %shared : i32
  } do {
  ^bb0(%arg2: i32):
    %res = arith.constant 0 : i32
    scf.yield %res, %res : i32, i32
  }
  
  return
}
 