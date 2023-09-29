use crate::AstNode;

#[derive(Debug, Clone)]
pub struct File {
    pub name: String,
    pub expr: ExprId,
    pub loc: Location,
}

#[derive(Debug, Clone, AstNode)]
#[visit(visit_expr)]
pub enum Expr {
    Fn(FnExpr),
    If(IfExpr),
    Let(LetExpr),
    Bin(BinExpr),
    Lit(LitExpr),
    Var(VarExpr),
    Call(CallExpr),
    Builtin(BuiltinExpr),
}

#[derive(Debug, Clone, AstNode)]
#[visit(visit_bin_expr)]
#[visit_skip(op, loc)]
pub struct BinExpr {
    pub op: BinOp,
    pub lhs: ExprId,
    pub rhs: ExprId,
    pub loc: Location,
}

#[derive(Debug, Clone, AstNode)]
#[visit(visit_var_expr)]
#[visit_skip_all]
pub struct VarExpr {
    pub ident: Ident,
    pub loc: Location,
}

#[derive(Debug, Clone, AstNode)]
#[visit(visit_fn_expr)]
#[visit_skip(params, loc)]
pub struct FnExpr {
    pub params: Vec<Param>,
    pub body: ExprId,
    pub loc: Location,
}

#[derive(Debug, Clone, AstNode)]
#[visit(visit_if_expr)]
#[visit_skip(loc)]
pub struct IfExpr {
    pub condition: ExprId,
    pub then: ExprId,
    pub otherwise: ExprId,
    pub loc: Location,
}

#[derive(Debug, Clone, AstNode)]
#[visit(visit_let_expr)]
#[visit_skip(name, loc)]
pub struct LetExpr {
    pub name: Param,
    pub init: ExprId,
    pub next: ExprId,
    pub loc: Location,
}

#[derive(Debug, Clone, AstNode)]
#[visit(visit_call_expr)]
#[visit_skip(loc)]
pub struct CallExpr {
    pub callee: ExprId,
    pub args: Vec<ExprId>,
    pub loc: Location,
}

#[derive(Debug, Clone, AstNode)]
#[visit(visit_builtin_expr)]
#[visit_skip(name, loc)]
pub struct BuiltinExpr {
    pub name: Ident,
    pub args: Vec<ExprId>,
    pub loc: Location,
}

#[derive(Debug, Clone, AstNode)]
#[visit(visit_lit_expr)]
#[visit_skip_all]
pub enum LitExpr {
    Str(StrLit),
    Int(IntLit),
    Bool(BoolLit),
    Tuple(TupleLit),
}

#[derive(Debug, Clone)]
pub struct StrLit {
    pub value: String,
    pub loc: Location,
}

#[derive(Debug, Clone)]
pub struct IntLit {
    pub value: i64,
    pub loc: Location,
}

#[derive(Debug, Clone)]
pub struct BoolLit {
    pub value: bool,
    pub loc: Location,
}

#[derive(Debug, Clone)]
pub struct TupleLit {
    pub first: ExprId,
    pub second: ExprId,
    pub loc: Location,
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    And,
    Or,
}

#[derive(Debug, Clone)]
pub struct Location(pub u32);

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct LocationData {
    pub start: usize,
    pub end: usize,
    pub file: InternedStr,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub ident: Ident,
    pub loc: Location,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ident(pub InternedStr);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExprId(pub u32);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct InternedStr(pub u32);
