
builtin.func @add(%old: i32, %new: i32) -> i32{
    %res = addi %old, %1 : i32
    return %res : i32
}

sdir.state @state_0{ 
    %a = sdir.get_access %A
    %c = sdir.get_access %C
    sdir.copy{wcr=@add} %a -> %c
}
