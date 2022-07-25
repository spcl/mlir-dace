// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK: sdfg.state
  sdfg.state @state_0{
    // CHECK: sdfg.nested_sdfg () -> ()
    sdfg.nested_sdfg () -> () {
      sdfg.state @state_0 {
      }
    }
  }
} 
