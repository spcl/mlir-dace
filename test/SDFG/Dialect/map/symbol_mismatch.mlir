// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: region with 1 blocks

sdfg.sdfg{entry=@state_0} {
    %A = sdfg.alloc() : !sdfg.array<2x6xi32>

    sdfg.state @state_0 {

        sdfg.map () = (0, 0) to (2, 2) step (1, 1) {
        }
    }
}
