// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        // CHECK: sdir.tasklet @five
        sdir.tasklet @five() -> index{
            %5 = arith.constant 5 : index
            sdir.return %5 : index
        }
        // CHECK: [[NAMEn:%[a-zA-Z0-9_]*]] = sdir.call @five()
        %n = sdir.call @five() : () -> index
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.alloc([[NAMEn]])
        // CHECK-SAME: !sdir.array<?x?xi32>
        %a = sdir.alloc(%n) : !sdir.array<?x?xi32>
    }
}
