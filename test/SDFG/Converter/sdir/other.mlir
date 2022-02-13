// XFAIL: *
// RUN: sdfg-opt --convert-to-sdfg %s
module{
    func @kernel_2mm(%ni: index) {   
        affine.for %i = 0 to %ni {
        }
        return
    }
}
