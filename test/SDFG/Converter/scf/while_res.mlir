// RUN: sdfg-opt --convert-to-sdfg %s
func.func private @main() -> i32 {
  %init1 = arith.constant 0 : i32

  %res = scf.while (%arg1 = %init1) : (i32) -> i32 {
    %c10 = arith.constant 10 : i32
    %condition = arith.cmpi sle, %arg1, %c10 : i32
    scf.condition(%condition) %arg1 : i32
  } do {
  ^bb0(%arg2: i32):
    %c1 = arith.constant 1 : i32
    %next = arith.addi %arg2, %c1 : i32
    scf.yield %next : i32
  }
  
  return %res : i32
}
 