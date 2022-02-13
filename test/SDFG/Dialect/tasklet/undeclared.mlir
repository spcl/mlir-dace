// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: use of undeclared SSA value name

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{

        %res = sdir.tasklet @add(%a: i32, %b: i32) -> i32 {
            %c = arith.addi %a, %b : i32
            sdir.return %c : i32
        }
    }
}
