// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: expected non-function type

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        sdir.tasklet @get_zero() -> {
            %c = arith.constant 0 : i32
            sdir.return %c : i32
        }
    }
}
