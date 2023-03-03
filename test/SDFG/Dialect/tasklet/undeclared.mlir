// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: use of undeclared SSA value name

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    %res = sdfg.tasklet(%a: i32, %b: i32) -> (i32) {
      %c = arith.addi %a, %b : i32
      sdfg.return %c : i32
    }
  }
}
