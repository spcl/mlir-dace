// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: block with no terminator

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    sdfg.tasklet() -> (i32) {
      %c = arith.constant 0 : i32
    }
  }
}
