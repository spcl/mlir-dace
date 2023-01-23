// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-translate --mlir-to-sdfg | python3 %S/../import_translation_test.py

// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-translate --mlir-to-sdfg | python3 %S/../execute_sdfg.py | FileCheck %s
// HACK: Has issues with parallel execution. Make sure it runs sequentially
// ALLOW_RETRIES: 10
// CHECK: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-NEXT: 42
// CHECK-NEXT: end_dump: [[ARRAY]]
func.func private @main(%out: memref<i32>){
  %c42 = arith.constant 42 : i32
  memref.store %c42, %out[] : memref<i32>
  return
}
