// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.alloc_transient
// CHECK-SAME: !sdir.memlet<i32>
%A = sdir.alloc_transient() : !sdir.memlet<i32>