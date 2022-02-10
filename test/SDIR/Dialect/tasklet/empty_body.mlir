// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: expected non-empty function body

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        sdir.tasklet @get_zero() -> i32{
        }
    }
}
