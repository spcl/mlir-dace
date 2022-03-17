// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: sdfg.sdfg 
sdfg.sdfg{} {
  // CHECK: sdfg.state {{@[a-zA-Z0-9_]*}}
  sdfg.state @state_0 {
    // CHECK: sdfg.nested_sdfg
    sdfg.nested_sdfg{} {
      // CHECK: sdfg.state 
      sdfg.state @state_1 {
      }
    }
  }
} 
