// XFAIL: *
// RUN: sdir-opt --convert-to-sdir %s
module{
    func @kernel_2mm(%ni: index) {   
        affine.for %i = 0 to %ni {
        }
        return
    }
}
