// RUN: sdir-opt %s | sdir-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = sdir.foo %{{.*}} : i32
        %res = sdir.foo %0 : i32
        return
    }
}