// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: func with invalid signature

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<2x6xi32>

    sdir.state @state_0 {

        sdir.map () = () to () step () {
            %B = sdir.load %A[0,0] : !sdir.array<2x6xi32> -> i32
        }
    }
}
