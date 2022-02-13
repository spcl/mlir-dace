// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: incorrect rank

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<2x12xi32>

    sdir.state @state_0 {
        %b = sdir.memlet_cast %A : !sdir.array<2x12xi32> -> !sdir.array<12xi32>
    }
} 
