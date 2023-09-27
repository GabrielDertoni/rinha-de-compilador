use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct File {
    pub name: String,
    #[serde(rename = "expression")]
    pub expr: Box<Expr>,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Expr {
    #[serde(rename = "Function")]
    Fn(FnExpr),
    If(IfExpr),
    Let(LetExpr),
    #[serde(rename = "Binary")]
    Bin(BinExpr),
    Var(VarExpr),
    Call(CallExpr),
    Str(StrLit),
    Int(IntLit),
    Tuple(TupleLit),
    First(FirstExpr),
    Second(SecondExpr),
    Print(PrintExpr),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinExpr {
    pub op: BinOp,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarExpr {
    #[serde(rename = "text")]
    pub ident: Ident,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnExpr {
    #[serde(rename = "parameters")]
    pub params: Vec<Param>,
    #[serde(alias = "value")]
    pub body: Box<Expr>,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IfExpr {
    pub condition: Box<Expr>,
    pub then: Box<Expr>,
    pub otherwise: Box<Expr>,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LetExpr {
    pub name: Param,
    #[serde(rename = "value")]
    pub init: Box<Expr>,
    pub next: Box<Expr>,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallExpr {
    pub callee: Box<Expr>,
    #[serde(rename = "arguments")]
    pub args: Vec<Box<Expr>>,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrLit {
    pub value: String,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntLit {
    pub value: i64,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TupleLit {
    pub first: Box<Expr>,
    pub second: Box<Expr>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrintExpr {
    pub value: Box<Expr>,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstExpr {
    pub value: Box<Expr>,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecondExpr {
    pub value: Box<Expr>,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinExpr {
    pub name: Ident,
    pub value: Box<Expr>,
    pub location: Location,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub start: usize,
    pub end: usize,
    #[serde(rename = "filename")]
    pub file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    #[serde(rename = "text")]
    pub ident: Ident,
    pub location: Location,
}

pub type Ident = String;

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
