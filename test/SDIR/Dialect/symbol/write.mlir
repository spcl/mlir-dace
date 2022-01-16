// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdir.state @state_0
    sdir.state @state_0{
        // CHECK-NEXT: sdir.alloc_symbol("N")
        sdir.alloc_symbol("N")
        // CHECK-NEXT: [[NAMEa:%[a-zA-Z0-9_]*]] = sdir.sym("3*N+2") : i64
        %1 = sdir.sym("3*N+2") : i64
        // CHECK-NEXT: sdir.sym_write [[NAMEa]], "N" : i64
        sdir.sym_write %1, "N" : i64
    }
}
