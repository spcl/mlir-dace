// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: processing elements is at least one

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.stream<2x6xi32>

    sdir.state @state_0 {

        sdir.consume{num_pes=-1} (%A : !sdir.stream<2x6xi32>) -> (pe: %p, elem: %e) {
        }
    }
}
