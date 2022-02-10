// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: starts with an alphabetical character

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        sdir.alloc_symbol("135")
    }
}
