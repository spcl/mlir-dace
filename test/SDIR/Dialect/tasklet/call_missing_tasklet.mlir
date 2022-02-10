// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: does not reference a valid tasklet or SDFG

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }

        %1 = sdir.call @one() : () -> i32
        %c = sdir.call @add(%1, %1) : (i32, i32) -> i32
    }
}
