// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: expected non-empty body

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    sdfg.tasklet() -> (i32) {
    }
  }
}
