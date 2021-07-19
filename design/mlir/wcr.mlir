
sdir.func @add(%old: i32, %new: i32) -> i32{
    %res = sdir.addi %old, %1 : i32
    sdir.return %res : i32
}

sdir.state @state_0{ 
    %a = sdir.get_access %A : !sdir.memlet<i32>
    %c = sdir.get_access %C : !sdir.memlet<i32>
    sdir.copy{wcr=@add} %a -> %c
}