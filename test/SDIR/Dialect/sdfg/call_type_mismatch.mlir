// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: type mismatch

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0 {
        sdir.sdfg{entry=@state_1} @sdfg_1(%a: !sdir.array<i64>, %b: !sdir.array<i32>) -> !sdir.array<i32> {
            sdir.state @state_1 {
            }
        }

        %N = sdir.alloc() : !sdir.array<i32>
        %c = sdir.call @sdfg_1(%N, %N) : (!sdir.array<i32>, !sdir.array<i32>) -> !sdir.array<i32>
    }
} 
