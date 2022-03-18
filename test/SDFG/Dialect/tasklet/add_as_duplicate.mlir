// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK: sdfg.state @state_0
  sdfg.state @state_0{
    // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: sdfg.tasklet
    %c = sdfg.tasklet() -> (i32) {
      %1 = arith.constant 1 : i32
      sdfg.return %1 : i32
    }

    // CHECK: sdfg.tasklet
    // CHECK-SAME: [[NAMEC]] as [[NAMEB1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[NAMEC]] as [[NAMEB2:%[a-zA-Z0-9_]*]]
    %s = sdfg.tasklet(%c as %b1: i32, %c as %b2: i32) -> (i32) {
      // CHECK-NEXT: [[NAMER:%[a-zA-Z0-9_]*]]
      // CHECK-SAME: [[NAMEB1]], [[NAMEB2]]
      %r = arith.addi %b1, %b2 : i32
      // CHECK-NEXT: sdfg.return [[NAMER]]
      sdfg.return %r : i32
    }
  }
}
