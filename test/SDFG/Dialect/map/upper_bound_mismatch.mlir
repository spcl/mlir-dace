// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: size of upper bounds matches size of arguments

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<2x6xi32>

    sdfg.state @state_0 {
        sdfg.map (%i, %j) = (0,0) to (2) step (1, 1) {
        }
    }
}
