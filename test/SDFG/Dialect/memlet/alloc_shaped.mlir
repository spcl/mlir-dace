// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdfg.alloc
  // CHECK-SAME: !sdfg.array<23x45x123xi32>
  %a = sdfg.alloc() : !sdfg.array<23x45x123xi32>

  sdfg.state @state_0{
  }
} 
