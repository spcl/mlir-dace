// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: parameter size matches undefined dimensions size

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    %n = sdfg.tasklet() -> (index) {
      %5 = arith.constant 5 : index
      sdfg.return %5 : index
    }

    %m = sdfg.tasklet() -> (index) {
      %20 = arith.constant 20 : index
      sdfg.return %20 : index
    }

    %a = sdfg.alloc(%n, %m) : !sdfg.array<?xi32>
  }
}
