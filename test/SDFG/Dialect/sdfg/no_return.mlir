// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: must return at least one value

sdfg.sdfg () -> () {
  sdfg.state @state_0{
  }
} 
