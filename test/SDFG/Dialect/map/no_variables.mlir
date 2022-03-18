// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: region with 1 blocks

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0 {
    sdfg.map () = () to () step () {
    }
  }
}
