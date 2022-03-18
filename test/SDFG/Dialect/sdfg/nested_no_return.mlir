// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: must return at least one value

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    sdfg.nested_sdfg () -> () {
      sdfg.state @state_0 {
      }
    }
  }
} 
