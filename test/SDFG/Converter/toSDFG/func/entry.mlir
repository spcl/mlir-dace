// RUN: sdfg-opt --convert-to-sdfg=main-func-name="f2" %s | sdfg-opt | FileCheck %s
// CHECK: arith.addi
func.func private @f1(%arg1: i32, %arg2: i32) -> i32 {
  %c0 = arith.subi %arg1, %arg2 : i32
  return %c0 : i32
}

func.func private @f2(%arg1: i32, %arg2: i32) -> i32 {
  %c0 = arith.addi %arg1, %arg2 : i32
  return %c0 : i32
}
