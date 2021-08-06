// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.alloc_stream()
// CHECK-SAME: !sdir.stream_array<i32>
%a = sdir.alloc_stream() : !sdir.stream_array<i32>

