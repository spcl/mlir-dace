#ifndef SDIR_StoreOp
#define SDIR_StoreOp

def SDIR_StoreOp : SDIR_Op<"store", [
    HasParent<"StateOp">,
    TypesMatchWith<"value type matches element type of 'memlet'",
                  "arr", "val",
                  "$_self.cast<MemletType>().getElementType()">
]> {
  let summary = "Memlet store operation";
  let description = [{
    Allows a state to store a value in a memlet.
    For example:

    ```mlir
    sdir.state @state_0 {
        ...
        %1 = constant 1 : i32
        %a = sdir.get_access %A : !sdir.memlet<i32>
        sdir.store %1, %a[0] : i32
        ...
    }
    ```
  }];

    let arguments = (ins SDIR_MemletType:$arr, I64Attr:$idx, AnyType:$val);
    let assemblyFormat = [{attr-dict $val `,` $arr `[` $idx `]` `:` type($arr)}];
}

#endif // SDIR_StoreOp
