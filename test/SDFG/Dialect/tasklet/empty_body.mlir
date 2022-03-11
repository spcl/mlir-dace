// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: expected non-empty body

sdfg.sdfg{entry=@state_0} {
    sdfg.state @state_0{
        sdfg.tasklet() -> (i32) {
        }
    }
}
