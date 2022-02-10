// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: requires a 'src' symbol reference attribute

sdir.sdfg @sdfg_0 {
    sdir.state @state_0{
    }

    sdir.state @state_1{
    }

    sdir.edge{assign=["i: 1"], condition=""} @state_0 -> @state_1
}
