// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: processing elements is at least one

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.stream<2x6xi32>

  sdfg.state @state_0 {
    sdfg.consume{num_pes=0} (%A : !sdfg.stream<2x6xi32>) -> (pe: %p, elem: %e) {
    }
  }
}
