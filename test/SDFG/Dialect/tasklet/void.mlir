// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK: sdfg.state
  sdfg.state @state_0{
    // CHECK: sdfg.tasklet () -> ()
    sdfg.tasklet () -> () {
      // CHECK: sdfg.return
      sdfg.return 
    }
  }
}
