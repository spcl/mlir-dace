// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: steps matches size of arguments

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0 {
    sdfg.map (%i, %j) = (0,0) to (2, 2) step (1) {
    }
  }
}
