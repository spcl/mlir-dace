// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: parameter size matches undefined dimensions size

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        %n = sdir.tasklet @five() -> index{
            %5 = arith.constant 5 : index
            sdir.return %5 : index
        }

        %m = sdir.tasklet @twenty() -> index{
            %20 = arith.constant 20 : index
            sdir.return %20 : index
        }

        %a = sdir.alloc(%n, %m) : !sdir.array<?xi32>
    }
}
