// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: processing elements is at least one

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.stream<2x6xi32>

    sdfg.state @state_0 {

        sdfg.consume{num_pes=-1} (%A : !sdfg.stream<2x6xi32>) -> (pe: %p, elem: %e) {
        }
    }
}
