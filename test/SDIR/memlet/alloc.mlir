// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.alloc
// CHECK-SAME: !sdir.memlet<i32>
%a = sdir.alloc() : !sdir.memlet<i32>
