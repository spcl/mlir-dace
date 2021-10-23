// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEn:%[a-zA-Z0-9_]*]]
    %n = constant 5 : index
    // CHECK-NEXT: [[NAMEm:%[a-zA-Z0-9_]*]]
    %m = constant 20 : index
    // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.alloc([[NAMEn]], [[NAMEm]])
    // CHECK-SAME: !sdir.array<?x6x?xi32>
    %a = sdir.alloc(%n, %m) : !sdir.array<?x6x?xi32>

    sdir.state @state_0{

    }
}
