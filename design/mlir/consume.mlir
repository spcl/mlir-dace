%A = sdir.alloc_stream() 
            : !sdir.stream<i32>

sdir.func @empty(%A : !sdir.stream<i32>)
                                    -> i1
{
    %l = sdir.stream_length %A : i32
    %isZero = sdir.cmpi "eq", %l, %0 : i32
    sdir.cond_br %isZero, ^zero, ^one
    
    ^zero: sdir.return %1 : i1
    ^one: sdir.return %0 : i1
}

sdir.consume{num_pes=$P, condition=@empty} 
    (%A: !sdir.stream<i32>) 
{
    ^bb0(%p: i32, %a: i32):
        %c = sdir.call @add_one(%a)
        sdir.store{wcr="add"} %c, %C[%0]
}
