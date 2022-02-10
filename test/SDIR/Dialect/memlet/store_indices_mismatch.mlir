// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: incorrect number of indices

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<i32>

    sdir.state @state_0 {
        sdir.tasklet @zero() -> index{
            %0 = arith.constant 0 : index
            sdir.return %0 : index
        }

        sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }

        %0 = sdir.call @zero() : () -> index
        %1 = sdir.call @one() : () -> i32

        sdir.store %1, %A[%0] : i32 -> !sdir.array<i32>
    }
} 
