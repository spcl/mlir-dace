// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: does not reference a valid tasklet or SDFG

sdfg.sdfg{entry=@state_0} @sdfg_0(%a: !sdfg.array<i32>, %b: !sdfg.array<i32>) -> !sdfg.array<i32> {
    sdfg.state @state_0 {
        sdfg.sdfg{entry=@state_1} @sdfg_1 {
            sdfg.state @state_1 {
                %N = sdfg.alloc() : !sdfg.array<i32>
                %c = sdfg.call @sdfg_0(%N, %N) : (!sdfg.array<i32>, !sdfg.array<i32>) -> !sdfg.array<i32>
            }
        }
    }
} 