// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: region entry argument '%c' is already in use

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    %c = sdfg.tasklet() -> (i32) {
      %1 = arith.constant 1 : i32
      sdfg.return %1 : i32
    }

    %s = sdfg.tasklet(%c: i32, %c: i32) -> (i32) {
      %r = arith.addi %c, %c : i32
      sdfg.return %r : i32
    }
  }
}
