// RUN: sdir-opt %s | sdir-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        // CHECK: %{{.*}} = sdir.state {label = "state_0"} : i32
        %res = sdir.state {label = "state_0"} : i32
        return
    }
}