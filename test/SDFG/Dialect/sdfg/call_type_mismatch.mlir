// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: expects different type

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0 {
    %N = sdfg.alloc() : !sdfg.array<i32>
    %R = sdfg.alloc() : !sdfg.array<i32>
    sdfg.nested_sdfg{entry=@state_1} (%N: !sdfg.array<i32>) -> (%R: !sdfg.array<i64>) {
      sdfg.state @state_1 {
      }
    }
  }
} 
