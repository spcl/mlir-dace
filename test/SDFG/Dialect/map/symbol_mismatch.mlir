// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: expected non-empty body

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0 {
    sdfg.map () = (0, 0) to (2, 2) step (1, 1) {
    }
  }
}
