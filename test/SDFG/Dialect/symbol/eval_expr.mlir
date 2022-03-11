// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK: sdfg.state @state_0
    sdfg.state @state_0{
        // CHECK-NEXT: sdfg.alloc_symbol("N")
        sdfg.alloc_symbol("N")
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdfg.sym("3*N+2") : i64
        %res = sdfg.sym("3*N+2") : i64
    }
}
