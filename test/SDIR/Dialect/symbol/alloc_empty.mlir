// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: string is not empty

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{
        sdir.alloc_symbol("")
    }
}
