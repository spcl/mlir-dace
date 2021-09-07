%A = sdir.alloc_stream() 
            : !sdir.stream<i32>

builtin.func @empty(%A : !sdir.stream<i32>)
                                    -> i1
{
    %l = sdir.stream_length %A : i32
    %isZero = cmpi "eq", %l, %0 : i32
    cond_br %isZero, ^zero, ^one
    
    ^zero: sdir.return %1 : i1
    ^one: sdir.return %0 : i1
}

sdir.consume{num_pes=5, condition=@empty} 
    (%A: !sdir.stream<i32>) 
                    -> (pe: %p, elem: %e)
{
    %c = sdir.call @add_one(%e)
    sdir.store{wcr="add"} %c, %C[0]
}
