// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: region with 1 blocks

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<2x6xi32>

    sdir.state @state_0 {

        sdir.map () = () to () step () {
        }
    }
}
