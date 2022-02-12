// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: use of undeclared SSA value name

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        %c = sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }

        %s = sdir.tasklet @add(%c as %b1: i32) -> i32{
            %r = arith.addi %c, %c : i32
            sdir.return %r : i32
        }
    }
}
