// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
// CHECK-SAME: {{@[a-zA-Z0-9_]*}}
sdfg.sdfg{entry=@state_0} {
  // CHECK: sdfg.state {{@[a-zA-Z0-9_]*}}
  sdfg.state @state_0{
    // CHECK: sdfg.nested_sdfg
    // CHECK-SAME: {{@[a-zA-Z0-9_]*}}
    sdfg.nested_sdfg{entry=@state_0} {
      // CHECK: sdfg.state {{@[a-zA-Z0-9_]*}}
      sdfg.state @state_0{
      }
    }
  }
} 
