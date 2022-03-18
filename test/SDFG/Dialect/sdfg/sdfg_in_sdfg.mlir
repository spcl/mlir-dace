// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: expects parent op 'builtin.module'

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
      sdfg.state @state_0{
      }
    }
  }
} 
