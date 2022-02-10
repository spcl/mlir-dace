// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: expects parent op 'sdir.tasklet'

builtin.func @add(%a: i32, %b: i32) -> i32{
    %c = arith.addi %a, %b : i32
    sdir.return %c : i32
}
