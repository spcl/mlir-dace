// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: block with no terminator

sdfg.sdfg{entry=@state_0} {
    sdfg.state @state_0{
        sdfg.tasklet @get_zero() -> i32{
            %c = arith.constant 0 : i32
        }
    }
}
