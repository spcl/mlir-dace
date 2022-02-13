// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: does not reference a valid tasklet or SDFG

sdir.sdfg{entry=@state_0} @sdfg_0(%a: !sdir.array<i32>, %b: !sdir.array<i32>) -> !sdir.array<i32> {
    sdir.state @state_0 {
        sdir.sdfg{entry=@state_1} @sdfg_1 {
            sdir.state @state_1 {
                %N = sdir.alloc() : !sdir.array<i32>
                %c = sdir.call @sdfg_0(%N, %N) : (!sdir.array<i32>, !sdir.array<i32>) -> !sdir.array<i32>
            }
        }
    }
} 
