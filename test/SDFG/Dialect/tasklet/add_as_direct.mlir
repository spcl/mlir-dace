// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: use of undeclared SSA value name

sdfg.sdfg{entry=@state_0} {
    sdfg.state @state_0{
        %c = sdfg.tasklet() -> i32{
            %1 = arith.constant 1 : i32
            sdfg.return %1 : i32
        }

        %s = sdfg.tasklet(%c as %b1: i32) -> i32{
            %r = arith.addi %c, %c : i32
            sdfg.return %r : i32
        }
    }
}
