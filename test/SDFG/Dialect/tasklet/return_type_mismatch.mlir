// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: 'sdir.return' op must match tasklet return types

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{

        %res = sdir.tasklet @get_zero() -> i32{
            %c = arith.constant 0 : i64
            sdir.return %c : i64
        }
    }
}
