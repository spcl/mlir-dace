#ifndef SDIR_LoadOp
#define SDIR_LoadOp

def SDIR_LoadOp : SDIR_Op<"load", [
    HasParent<"StateOp">,
    TypesMatchWith<"result type matches element type of 'memlet'",
                     "arr", "res",
                     "$_self.cast<MemletType>().getElementType()">
]> {
  let summary = "Memlet load operation";
  let description = [{
    Allows a state to load a value from a memlet.
    For example:

    ```mlir
    sdir.state @state_0 {
        ...
        %a = sdir.get_access %A : !sdir.memlet<i32>
        %a_1 = sdir.load %a[0] : i32
        ...
    }
    ```
  }];

    let arguments = (ins SDIR_MemletType:$arr, I64Attr:$idx);
    let results = (outs AnyType:$res);
    let assemblyFormat = [{attr-dict $arr `[` $idx `]` `:` type($arr)}];
}

#endif // SDIR_LoadOp