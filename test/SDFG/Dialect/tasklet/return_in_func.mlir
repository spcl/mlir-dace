// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: expects parent op 'sdfg.tasklet'

builtin.func @add(%a: i32, %b: i32) -> i32{
  %c = arith.addi %a, %b : i32
  sdfg.return %c : i32
}
