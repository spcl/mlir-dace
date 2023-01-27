// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: sdfg.sdfg 
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK: sdfg.state [[State0:@[a-zA-Z0-9_]*]] 
  sdfg.state @state_0{
  }

  // CHECK: sdfg.state [[State1:@[a-zA-Z0-9_]*]] 
  sdfg.state @state_1{
  }

  // CHECK: sdfg.edge {assign = [], condition = "1"} [[State0]] -> [[State1]]
  sdfg.edge @state_0 -> @state_1
}
