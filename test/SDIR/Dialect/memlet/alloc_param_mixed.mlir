// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        // CHECK: [[NAMEn:%[a-zA-Z0-9_]*]] = sdir.tasklet @five
        %n = sdir.tasklet @five() -> index{
            %5 = arith.constant 5 : index
            sdir.return %5 : index
        }

        // CHECK: [[NAMEm:%[a-zA-Z0-9_]*]] = sdir.tasklet @twenty
        %m = sdir.tasklet @twenty() -> index{
            %20 = arith.constant 20 : index
            sdir.return %20 : index
        }
        // CHECK: {{%[a-zA-Z0-9_]*}} = sdir.alloc([[NAMEn]], [[NAMEm]])
        // CHECK-SAME: !sdir.array<?x6x?xi32>
        %a = sdir.alloc(%n, %m) : !sdir.array<?x6x?xi32>
    }
}
