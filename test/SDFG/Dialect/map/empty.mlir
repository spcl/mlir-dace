// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK: sdfg.state
  // CHECK-SAME: @state_0
  sdfg.state @state_0 {
    // CHECK: sdfg.map
    sdfg.map (%i, %j) = (0, 0) to (2, 2) step (1, 1) {
    }
  }
}
