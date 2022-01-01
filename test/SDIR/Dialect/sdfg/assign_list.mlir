// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdir.state [[STATE0:@[a-zA-Z0-9_]*]]
    sdir.state @state_0{

    }

    // CHECK: sdir.state [[STATE1:@[a-zA-Z0-9_]*]]
    sdir.state @state_1{

    }

    // CHECK: sdir.edge
    // CHECK-SAME: assign = ["i = 1", "j = 5"]
    // CHECK-SAME: [[STATE0]] -> [[STATE1]]
    sdir.edge{assign=["i = 1", "j = 5"], condition=""} @state_0 -> @state_1
}
