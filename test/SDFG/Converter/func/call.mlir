// XFAIL: *
// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt
func.func private @ex(i32, i32)

func.func private @main(%arg1: i32, %arg2: i32) {
  func.call @ex(%arg1, %arg2) : (i32, i32) -> ()
  return
}
