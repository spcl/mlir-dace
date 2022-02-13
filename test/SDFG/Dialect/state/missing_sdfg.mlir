// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: expects parent op 'sdfg.sdfg'

sdfg.state @state_0{
}
