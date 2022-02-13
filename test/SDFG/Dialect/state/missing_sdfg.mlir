// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: expects parent op 'sdir.sdfg'

sdir.state @state_0{
}
