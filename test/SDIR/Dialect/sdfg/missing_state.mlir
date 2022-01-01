// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdir.state [[STATE0:@[a-zA-Z0-9_]*]]
    sdir.state @state_0{

    }

    // CHECK: sdir.edge
    // CHECK-SAME: [[STATE0]]
    sdir.edge{assign=["i: 1"], condition=""} @state_0 -> @state_1
}
