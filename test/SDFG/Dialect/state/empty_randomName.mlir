// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@randomName} @sdfg_0 {
    // CHECK: sdfg.state @randomName
    sdfg.state @randomName{

    }
}
