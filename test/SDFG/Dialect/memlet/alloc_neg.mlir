// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: parameter size matches undefined dimensions size

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %a = sdfg.alloc() : !sdfg.array<-1xi32>

    sdfg.state @state_0{
    }
}
