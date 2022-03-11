// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: parameter size matches undefined dimensions size

sdfg.sdfg{entry=@state_0} {
    sdfg.state @state_0{
        %n = sdfg.tasklet @five() -> index{
            %5 = arith.constant 5 : index
            sdfg.return %5 : index
        }

        %m = sdfg.tasklet @twenty() -> index{
            %20 = arith.constant 20 : index
            sdfg.return %20 : index
        }

        %a = sdfg.alloc(%n, %m) : !sdfg.array<?xi32>
    }
}
