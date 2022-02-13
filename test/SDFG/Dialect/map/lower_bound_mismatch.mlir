// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: size of lower bounds matches size of arguments

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<2x6xi32>

    sdir.state @state_0 {

        sdir.map (%i, %j) = (0) to (2, 2) step (1, 1) {
        }
    }
}
