// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-translate --mlir-to-sdfg | python3 %S/../execute_sdfg.py | FileCheck %s
// CHECK: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-NEXT: 3
// CHECK-NEXT: end_dump: [[ARRAY]]

// CHECK-NEXT: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-NEXT: 3
// CHECK-NEXT: end_dump: [[ARRAY]]

// CHECK-NEXT: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-NEXT: 6
// CHECK-NEXT: end_dump: [[ARRAY]]
func.func private @main(%arg1: i32, %arg2: i32) -> i32 {
  %c0 = arith.addi %arg1, %arg2 : i32
  return %c0 : i32
}
