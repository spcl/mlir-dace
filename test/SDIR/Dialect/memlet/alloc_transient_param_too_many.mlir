// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEn:%[a-zA-Z0-9_]*]]
    %n = constant 5 : i32
    // CHECK-NEXT: [[NAMEm:%[a-zA-Z0-9_]*]]
    %m = constant 20 : i32
    // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.alloc_transient([[NAMEn]], [[NAMEm]])
    // CHECK-SAME: !sdir.array<?xi32>
    %a = sdir.alloc_transient(%n, %m) : !sdir.array<?xi32>

    sdir.state @state_0{

    }
}
