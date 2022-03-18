// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: parameter size matches undefined dimensions size

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %a = sdfg.alloc() : !sdfg.array<-1xi32>

  sdfg.state @state_0{
  }
}
