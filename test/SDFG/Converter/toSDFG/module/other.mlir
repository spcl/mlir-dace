// RUN: not sdfg-opt --convert-to-sdfg %s 2>&1 | FileCheck %s
// CHECK: failed to legalize operation 'affine.for'

module{
    func.func @main(%ni: index) {   
        affine.for %i = 0 to %ni {
        }
        return
    }
}
