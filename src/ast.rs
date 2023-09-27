#[derive(Debug, Clone)]
pub struct File {
    pub name: String,
    pub expr: ExprId,
    pub loc: Location,
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct BinExpr {
    pub op: BinOp,
    pub lhs: ExprId,
    pub rhs: ExprId,
    pub loc: Location,
}

#[derive(Debug, Clone)]
pub struct VarExpr {
    pub ident: Ident,
    pub loc: Location,
}

#[derive(Debug, Clone)]
pub struct FnExpr {
    pub params: Vec<Param>,
    pub body: ExprId,
    pub loc: Location,
}

#[derive(Debug, Clone)]
pub struct IfExpr {
    pub condition: ExprId,
    pub then: ExprId,
    pub otherwise: ExprId,
    pub loc: Location,
}

#[derive(Debug, Clone)]
pub struct LetExpr {
    pub name: Param,
    pub init: ExprId,
    pub next: ExprId,
    pub loc: Location,
}

#[derive(Debug, Clone)]
pub struct CallExpr {
    pub callee: ExprId,
    pub args: Vec<ExprId>,
    pub loc: Location,
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct BuiltinExpr {
    pub name: Ident,
    pub args: Vec<ExprId>,
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
pub struct Location {
    pub start: usize,
    pub end: usize,
    pub file: String,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub ident: Ident,
    pub loc: Location,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ident(pub u32);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExprId(pub u32);

/* -- Impls -- */

impl Location {
    pub fn builtin() -> Location {
        Location {
            start: 0,
            end: 0,
            file: String::from("builtin"),
        }
    }
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Rem => write!(f, "%"),
            BinOp::Eq => write!(f, "=="),
            BinOp::Neq => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Lte => write!(f, "<="),
            BinOp::Gte => write!(f, ">="),
            BinOp::And => write!(f, "and"),
            BinOp::Or => write!(f, "or"),
        }
    }
}
