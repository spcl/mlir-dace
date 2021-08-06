// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg @sdfg_0 {
    // CHECK-NEXT: sdir.state
    // CHECK-DAG: nosync = false
    // CHECK-DAG: instrument = "No_Instrumentation"
    // CHECK-SAME: @state_0
    sdir.state {
        nosync=false,
        instrument="No_Instrumentation"
    } @state_0 {

    }
}