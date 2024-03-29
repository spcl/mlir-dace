// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdfg.alloc
  // CHECK-SAME: !sdfg.array<i32>
  %a = sdfg.alloc() : !sdfg.array<i32>

  sdfg.state @state_0{
  }
}
