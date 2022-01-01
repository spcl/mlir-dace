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
        // CHECK: sdir.tasklet @twenty
        sdir.tasklet @twenty() -> index{
            %20 = arith.constant 20 : index
            sdir.return %20 : index
        }
        // CHECK: [[NAMEn:%[a-zA-Z0-9_]*]] = sdir.call @five()
        %n = sdir.call @five() : () -> index
        // CHECK-NEXT: [[NAMEm:%[a-zA-Z0-9_]*]] = sdir.call @twenty()
        %m = sdir.call @twenty() : () -> index
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.alloc_transient([[NAMEn]], [[NAMEm]])
        // CHECK-SAME: !sdir.array<?xi32>
        %a = sdir.alloc_transient(%n, %m) : !sdir.array<?xi32>
    }
}
