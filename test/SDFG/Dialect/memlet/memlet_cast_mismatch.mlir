// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: incorrect rank

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<2x12xi32>

    sdfg.state @state_0 {
        %b = sdfg.memlet_cast %A : !sdfg.array<2x12xi32> -> !sdfg.array<12xi32>
    }
} 
