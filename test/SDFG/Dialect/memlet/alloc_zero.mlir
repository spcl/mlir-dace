// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: dimensions of size zero

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %a = sdfg.alloc() : !sdfg.array<0xi32>

  sdfg.state @state_0{
  }
}
