// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
// CHECK-SAME: {{@[a-zA-Z0-9_]*}}
sdfg.sdfg{entry=@state_0} {
  // CHECK: sdfg.state {{@[a-zA-Z0-9_]*}}
  sdfg.state @state_0{
    // CHECK: [[ARRAYN:%[a-zA-Z0-9_]*]] = sdfg.alloc
    // CHECK-SAME: !sdfg.array<i32>
    %N = sdfg.alloc() : !sdfg.array<i32>
    // CHECK: sdfg.nested_sdfg
    // CHECK-SAME: {{@[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARRAYN]]
    sdfg.nested_sdfg{entry=@state_0} () -> (%N: !sdfg.array<i32>){
      // CHECK: sdfg.state {{@[a-zA-Z0-9_]*}}
      sdfg.state @state_0{
      }
    }
  }
} 
