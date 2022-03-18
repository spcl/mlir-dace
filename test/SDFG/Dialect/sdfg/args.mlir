// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: sdfg.sdfg 
// CHECK-SAME: {{%[a-zA-Z0-9_]*}}: !sdfg.array<i32>
// CHECK-SAME: ->
// CHECK-SAME: {{%[a-zA-Z0-9_]*}}: !sdfg.array<i32>
sdfg.sdfg {entry = @state_0}(%arg0: !sdfg.array<i32>) -> (%arg1: !sdfg.array<i32>) {
  sdfg.state @state_0 {
  }
}

