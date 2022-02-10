// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: dimensions of size zero

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %a = sdir.alloc() : !sdir.array<0xi32>

    sdir.state @state_0{

    }
}
