// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK-LABEL: sdir.state {instrument = "No_Instrumentation", nosync = false} @state_0
sdir.state{instrument="No_Instrumentation", nosync=false} @state_0{

}
