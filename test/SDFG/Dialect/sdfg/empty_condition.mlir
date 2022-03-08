// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: condition must be non-empty or omitted

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    sdfg.state @state_0{}
    sdfg.state @state_1{}
    sdfg.edge{assign=["i: 1"], condition=""} @state_0 -> @state_1
}
