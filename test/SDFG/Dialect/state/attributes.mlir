// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK: sdfg.state
    // CHECK-DAG: nosync = false
    // CHECK-DAG: instrument = "No_Instrumentation"
    // CHECK-SAME: @state_0
    sdfg.state {
        nosync=false,
        instrument="No_Instrumentation"
    } @state_0 {

    }
}
