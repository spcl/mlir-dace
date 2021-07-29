// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// Module is always implicitly top-level
// CHECK: module
// CHECK-NEXT: sdir.state
// CHECK-DAG: nosync = false
// CHECK-DAG: instrument = "No_Instrumentation"
// CHECK-SAME: @state_0
sdir.state {
    nosync=false,
    instrument="No_Instrumentation"
} @state_0 {

}
