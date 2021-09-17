// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdir.state [[STATE0:@[a-zA-Z0-9_]*]]
    sdir.state @state_0{
        // CHECK: sdir.sdfg
        // CHECK-SAME: {{@[a-zA-Z0-9_]*}}
        sdir.sdfg{entry=@state_1} @sdfg_1 {
            // CHECK: sdir.state [[STATE2:@[a-zA-Z0-9_]*]]
            sdir.state @state_2{

            }
            // CHECK: sdir.edge
            // CHECK-SAME: [[STATE2]] -> [[STATE0]]
            sdir.edge{assign=["i = 1"], condition=""} @state_2 -> @state_0
        }
    }
} 
