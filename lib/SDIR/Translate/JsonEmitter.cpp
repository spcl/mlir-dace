#include "mlir/IR/Identifier.h"
#include "mlir/IR/Types.h"
#include "SDIR/Translate/JsonEmitter.h"

using namespace mlir;

JsonEmitter::JsonEmitter(raw_ostream &os): os(os) {
    indentation = 0;
    error = false;
    firstEntry = true;
    emptyLine = true;
    symStack.clear();
}

bool JsonEmitter::finish(){
    while(!symStack.empty()){
        SYM sym = symStack.pop_back_val();
        newLine();
        unindent();
        if(sym == SYM::BRACE) printLiteral("}");
        if(sym == SYM::SQUARE) printLiteral("]");
        os.changeColor(os.RED, /*Bold=*/true);
        printLiteral(" <<<<<<<<<<<< Auto-inserted closing clause");
        os.resetColor();
        error = true;
    }
    newLine(); // Makes sure to have a trailing newline
    return error;
}

void JsonEmitter::indent(){ indentation += 2; }
void JsonEmitter::unindent(){ indentation = indentation >= 2 ? indentation-2 : 0; }

void JsonEmitter::newLine(){
    if(emptyLine) return;
    os << "\n";
    emptyLine = true;
}

void JsonEmitter::printLiteral(StringRef str){
    if(emptyLine) os.indent(indentation);
    os << str;
    emptyLine = false;
}

void JsonEmitter::printString(StringRef str){
    printLiteral("\"");
    printLiteral(str);
    printLiteral("\"");
}

void JsonEmitter::startObject(){
    startEntry();
    printLiteral("{");
    symStack.push_back(SYM::BRACE);
    indent();
    newLine();
    firstEntry = true;
}

void JsonEmitter::startNamedObject(StringRef name){
    startEntry();
    printString(name);
    printLiteral(": ");
    printLiteral("{");
    symStack.push_back(SYM::BRACE);
    indent();
    newLine();
    firstEntry = true;
}

void JsonEmitter::endObject(){
    newLine();
    unindent();
    tryPop(SYM::BRACE);
    printLiteral("}");
    firstEntry = false;
}

void JsonEmitter::startEntry(){
    if(!firstEntry) printLiteral(",");
    firstEntry = false;
    newLine();
}

void JsonEmitter::printKVPair(StringRef key, StringRef val, bool stringify){
    startEntry();
    printString(key);
    printLiteral(": ");
    if(stringify) printString(val);
    else printLiteral(val);
}

void JsonEmitter::printKVPair(StringRef key, int val, bool stringify){
    startEntry();
    printString(key);
    printLiteral(": ");
    if(stringify) printLiteral("\"");
    os << val;
    if(stringify) printLiteral("\"");
}

void JsonEmitter::printKVPair(StringRef key, Attribute &val, bool stringify){
    startEntry();
    printString(key);
    printLiteral(": ");
    if(StringAttr strAttr = val.dyn_cast<StringAttr>()){
        strAttr.print(os);
    } else {
        if(stringify) printLiteral("\"");
        val.print(os);
        if(stringify) printLiteral("\"");
    }
}

void JsonEmitter::printKVPair(StringRef key, Region &val, bool stringify){
    startEntry();
    printString(key);
    printLiteral(": ");
    if(stringify) printLiteral("\"");
    for(Operation &oper : val.getOps()){
        oper.print(os);
        printLiteral("\\n");
    }
    if(stringify) printLiteral("\"");
}

void JsonEmitter::startNamedList(StringRef name){
    startEntry();
    printString(name);
    printLiteral(": ");
    printLiteral("[");
    symStack.push_back(SYM::SQUARE);
    indent();
    newLine();
    firstEntry = true;
}

void JsonEmitter::endList(){
    newLine();
    unindent();
    tryPop(SYM::SQUARE);
    printLiteral("]");
    firstEntry = false;
}

void JsonEmitter::tryPop(SYM sym){
    if(symStack.empty()){
        if(sym == SYM::BRACE) printLiteral("{");
        if(sym == SYM::SQUARE) printLiteral("[");
        os.changeColor(os.RED, /*Bold=*/true);
        printLiteral(" <<<<<<<<<<<< Auto-inserted opening clause");
        os.resetColor();
        newLine();
        indent();
        error = true;
    } else if(symStack.back() != sym){
        if(sym == SYM::BRACE) printLiteral("]");
        if(sym == SYM::SQUARE) printLiteral("}");
        os.changeColor(os.RED, /*Bold=*/true);
        printLiteral(" <<<<<<<<<<<< Auto-inserted closing clause");
        os.resetColor();
        newLine();
        unindent();
        error = true;
        symStack.pop_back();
    } else {
        symStack.pop_back();
    }
}

void JsonEmitter::printAttributes(ArrayRef<NamedAttribute> arr, 
                      ArrayRef<StringRef> elidedAttrs){ 
    
    llvm::SmallDenseSet<StringRef> elidedAttrsSet(elidedAttrs.begin(),
                                                elidedAttrs.end());
    
    for(NamedAttribute attr : arr){
        if(elidedAttrsSet.contains(attr.first.strref())) continue;
        printKVPair(attr.first, attr.second);
    }      
}