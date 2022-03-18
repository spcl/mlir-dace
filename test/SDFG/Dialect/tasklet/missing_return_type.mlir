// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: expected non-function type

sdfg.sdfg{entry=@state_0} {
    sdfg.state @state_0{
        sdfg.tasklet () -> {
            %c = arith.constant 0 : i32
            sdfg.return %c : i32
        }
    }
}
