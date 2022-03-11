// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK: sdfg.state @state_0
    sdfg.state @state_0{
        // CHECK-NEXT: sdfg.alloc_symbol ("N4")
        sdfg.alloc_symbol("N4")
    }
}
