// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
// CHECK-SAME: {{@[a-zA-Z0-9_]*}}
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdir.state {{@[a-zA-Z0-9_]*}}
    sdir.state @state_0{
        // CHECK: sdir.sdfg
        // CHECK-SAME: {{@[a-zA-Z0-9_]*}}
        sdir.sdfg{entry=@state_0} @sdfg_0 {
            // CHECK: sdir.state {{@[a-zA-Z0-9_]*}}
            sdir.state @state_0{

            }
        }
    }
} 
