// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK: sdfg.state [[STATE0:@[a-zA-Z0-9_]*]]
  sdfg.state @state_0{
  }

  // CHECK: sdfg.state [[STATE1:@[a-zA-Z0-9_]*]]
  sdfg.state @state_1{
  }

  // CHECK: sdfg.edge
  // CHECK-SAME: condition = "i > 1"
  // CHECK-SAME: [[STATE0]] -> [[STATE1]]
  sdfg.edge{condition="i > 1"} @state_0 -> @state_1
}
