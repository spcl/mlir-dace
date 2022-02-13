// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: starts with an alphabetical character

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    sdfg.state @state_0{
        sdfg.alloc_symbol("135")
    }
}
