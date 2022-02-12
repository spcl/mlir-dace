// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: incorrect number of indices

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<12xi32>

    sdir.state @state_0 {
        %0 = sdir.tasklet @zero() -> index{
            %0 = arith.constant 0 : index
            sdir.return %0 : index
        }
        
        %a_1 = sdir.load %A[%0, %0] : !sdir.array<12xi32> -> i32
    }
} 
