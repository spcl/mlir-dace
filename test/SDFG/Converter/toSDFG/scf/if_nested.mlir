// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt
func.func private @main(%arg0: i32, %arg1: i32) {
  %0 = arith.cmpi ne, %arg0, %arg1 : i32
  %1 = arith.cmpi eq, %arg0, %arg1 : i32
  scf.if %0 {
    scf.if %1 {
      %c1_i32 = arith.constant 1 : i32
    }
  }
  return
}
