// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: expects parent op to be one of 'sdfg.sdfg, sdfg.nested_sdfg'

sdfg.state @state_0{
}
