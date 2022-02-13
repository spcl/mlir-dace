// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: 'sdfg.return' op must match tasklet return types

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    sdfg.state @state_0{

        %res = sdfg.tasklet @get_zero() -> i32{
            %c = arith.constant 0 : i64
            sdfg.return %c : i64
        }
    }
}
