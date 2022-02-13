// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: type mismatch

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    sdfg.state @state_0 {
        sdfg.sdfg{entry=@state_1} @sdfg_1(%a: !sdfg.array<i64>, %b: !sdfg.array<i32>) -> !sdfg.array<i32> {
            sdfg.state @state_1 {
            }
        }

        %N = sdfg.alloc() : !sdfg.array<i32>
        %c = sdfg.call @sdfg_1(%N, %N) : (!sdfg.array<i32>, !sdfg.array<i32>) -> !sdfg.array<i32>
    }
} 
