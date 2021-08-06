// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: sdir.sdfg
sdir.sdfg @sdfg_0 {
    // CHECK: sdir.state @state_0
    sdir.state @state_0
}