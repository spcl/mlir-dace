// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: 'sdfg.return' op must match tasklet return types

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    %res = sdfg.tasklet() -> (i32) {
      %c = arith.constant 0 : i64
      sdfg.return %c : i64
    }
  }
}
