// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: 1 operands present, but expected 2

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        sdir.tasklet @add(%a: i32, %b: i32) -> i32{
            %c = arith.addi %a, %b : i32
            sdir.return %c : i32
        }

        sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }

        %1 = sdir.call @one() : () -> i32
        %c = sdir.call @add(%1) : (i32, i32) -> i32
    }
}
