// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-translate --mlir-to-sdfg | python3 %S/../import_translation_test.py

// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-translate --mlir-to-sdfg | python3 %S/../execute_sdfg.py | FileCheck %s
// HACK: Has issues with parallel execution. Make sure it runs sequentially
// ALLOW_RETRIES: 10
// CHECK: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-NEXT: 3
// CHECK-NEXT: end_dump: [[ARRAY]]

// CHECK-NEXT: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-NEXT: 3
// CHECK-NEXT: end_dump: [[ARRAY]]

// CHECK-NEXT: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-NEXT: 6
// CHECK-NEXT: end_dump: [[ARRAY]]
func.func private @main(%arg1: i32, %arg2: i32, %out: memref<i32>) {
  %c0 = arith.addi %arg1, %arg2 : i32
  memref.store %c0, %out[] : memref<i32>
  return
}
