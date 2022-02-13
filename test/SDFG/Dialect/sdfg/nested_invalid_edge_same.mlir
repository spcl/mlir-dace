// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: does not reference a valid state

sdir.sdfg{entry=@state_0} @sdfg_0 {
    sdir.state @state_0{

        sdir.sdfg{entry=@state_1} @sdfg_1 {
            sdir.state @state_2{
            }

            sdir.edge{assign=["i: 1"], condition=""} @state_2 -> @state_0
        }
    }
} 
