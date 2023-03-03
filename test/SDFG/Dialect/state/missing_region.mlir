// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: expected '{' to begin a region

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0
}
