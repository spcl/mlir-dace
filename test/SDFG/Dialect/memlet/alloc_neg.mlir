// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: parameter size matches undefined dimensions size

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %a = sdir.alloc() : !sdir.array<-1xi32>

    sdir.state @state_0{
    }
}
