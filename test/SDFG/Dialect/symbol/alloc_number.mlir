// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: starts with an alphabetical character

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    sdfg.alloc_symbol("135")
  }
}
