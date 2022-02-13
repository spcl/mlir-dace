// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdfg.state [[STATE0:@[a-zA-Z0-9_]*]]
    sdfg.state @state_0{

    }

    // CHECK: sdfg.state [[STATE1:@[a-zA-Z0-9_]*]]
    sdfg.state @state_1{

    }

    // CHECK: sdfg.edge
    // CHECK-SAME: [[STATE0]] -> [[STATE1]]
    sdfg.edge{assign=["i: 1"], condition=""} @state_0 -> @state_1
}
