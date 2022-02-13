// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: requires a 'src' symbol reference attribute

sdfg.sdfg @sdfg_0 {
    sdfg.state @state_0{
    }

    sdfg.state @state_1{
    }

    sdfg.edge{assign=["i: 1"], condition=""} @state_0 -> @state_1
}
