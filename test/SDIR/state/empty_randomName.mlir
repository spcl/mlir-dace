// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@randomName} @sdfg_0 {
    // CHECK: sdir.state @randomName
    sdir.state @randomName{

    }
}