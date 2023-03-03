// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: expected valid '@'-identifier

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state state_0{
  }
}
