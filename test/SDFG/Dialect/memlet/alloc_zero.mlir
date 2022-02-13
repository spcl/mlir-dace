// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: dimensions of size zero

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %a = sdfg.alloc() : !sdfg.array<0xi32>

    sdfg.state @state_0{

    }
}
