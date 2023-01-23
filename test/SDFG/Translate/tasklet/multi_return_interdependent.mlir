// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../execute_sdfg.py | FileCheck %s
// HACK: Has issues with parallel execution. Make sure it runs sequentially
// ALLOW_RETRIES: 10
// CHECK: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-NEXT: 1
// CHECK-NEXT: end_dump: [[ARRAY]]

// CHECK-NEXT: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-NEXT: 2
// CHECK-NEXT: end_dump: [[ARRAY]]

sdfg.sdfg () -> (%arg0: !sdfg.array<i32>, %arg1: !sdfg.array<i32>) {
  sdfg.state @state_0{
    %n:2 = sdfg.tasklet() -> (i32, i32) {
      %1 = arith.constant 1 : i32
      %2 = arith.addi %1, %1 : i32
      sdfg.return %1, %2 : i32, i32
    }

    sdfg.store %n#0, %arg0[] : i32 -> !sdfg.array<i32>
    sdfg.store %n#1, %arg1[] : i32 -> !sdfg.array<i32>
  }
}
