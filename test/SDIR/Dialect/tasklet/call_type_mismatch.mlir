// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: expects different type

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        sdir.tasklet @add(%a: i32, %b: i32) -> i32{
            %c = arith.addi %a, %b : i32
            sdir.return %c : i32
        }

        sdir.tasklet @one() -> i64{
            %1 = arith.constant 1 : i64
            sdir.return %1 : i64
        }

        %1 = sdir.call @one() : () -> i64
        %c = sdir.call @add(%1, %1) : (i32, i32) -> i32
    }
}
