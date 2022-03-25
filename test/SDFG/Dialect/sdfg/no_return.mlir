// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: sdfg.sdfg () -> ()
sdfg.sdfg () -> () {
  sdfg.state @state_0{
  }
} 
