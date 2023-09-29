use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Not, Rem, Sub};
use std::rc::Rc;

use crate::exception::ExceptionKind;
use crate::Env;

type OpResult = Result<Value, ExceptionKind>;

#[derive(Debug, Clone)]
pub enum Type {
    Str,
    Int,
    Bool,
    Tuple,
    Closure,
}

#[derive(Debug, Clone)]
pub enum Value {
    Str(String),
    Int(i64),
    Bool(bool),
    Tuple(Box<Value>, Box<Value>),
    Closure(Rc<Closure>),
}

impl Value {
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Str(s) => !s.is_empty(),
            &Value::Int(i) => i != 0,
            &Value::Bool(b) => b,
            Value::Tuple(_, _) => true,
            Value::Closure(_) => true,
        }
    }

    pub fn type_of(&self) -> Type {
        match self {
            Value::Str(_) => Type::Str,
            Value::Int(_) => Type::Int,
            Value::Bool(_) => Type::Bool,
            Value::Tuple(_, _) => Type::Tuple,
            Value::Closure(_) => Type::Closure,
        }
    }

    pub fn as_closure_mut(&mut self) -> Option<&mut Rc<Closure>> {
        if let Self::Closure(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Closure {
    pub capture: Env,
    pub func: ast::FnExpr,
    pub named: Option<ast::Ident>,
}

impl Add for &Value {
    type Output = OpResult;

    fn add(self, rhs: Self) -> Self::Output {
        let output = match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs + rhs),
            (Value::Str(lhs), Value::Str(rhs)) => {
                let mut tmp = lhs.clone();
                tmp.push_str(rhs.as_str());
                Value::Str(tmp)
            }
            (lhs, rhs) => {
                return Err(ExceptionKind::UnsupportedOperandTypes {
                    op: ast::BinOp::Add,
                    lhs: lhs.type_of(),
                    rhs: rhs.type_of(),
                })
            }
        };
        Ok(output)
    }
}

impl Add for Value {
    type Output = OpResult;

    fn add(self, rhs: Self) -> Self::Output {
        Add::add(&self, &rhs)
    }
}

impl Sub for &Value {
    type Output = OpResult;

    fn sub(self, rhs: Self) -> Self::Output {
        let output = match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs - rhs),
            (lhs, rhs) => {
                return Err(ExceptionKind::UnsupportedOperandTypes {
                    op: ast::BinOp::Sub,
                    lhs: lhs.type_of(),
                    rhs: rhs.type_of(),
                })
            }
        };
        Ok(output)
    }
}

impl Sub for Value {
    type Output = OpResult;

    fn sub(self, rhs: Self) -> Self::Output {
        Sub::sub(&self, &rhs)
    }
}

impl Mul for &Value {
    type Output = OpResult;

    fn mul(self, rhs: Self) -> Self::Output {
        let output = match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs * rhs),
            (Value::Str(lhs), &Value::Int(rhs)) => Value::Str(mul_str(lhs.as_str(), rhs)),
            (&Value::Int(lhs), Value::Str(rhs)) => Value::Str(mul_str(rhs.as_str(), lhs)),
            (lhs, rhs) => {
                return Err(ExceptionKind::UnsupportedOperandTypes {
                    op: ast::BinOp::Mul,
                    lhs: lhs.type_of(),
                    rhs: rhs.type_of(),
                })
            }
        };
        Ok(output)
    }
}

impl Mul for Value {
    type Output = OpResult;

    fn mul(self, rhs: Self) -> Self::Output {
        Mul::mul(&self, &rhs)
    }
}

fn mul_str(s: &str, times: i64) -> String {
    std::iter::repeat(s)
        .take(times.try_into().unwrap_or(0))
        .collect()
}

impl Div for &Value {
    type Output = OpResult;

    fn div(self, rhs: Self) -> Self::Output {
        let output = match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs / rhs),
            (lhs, rhs) => {
                return Err(ExceptionKind::UnsupportedOperandTypes {
                    op: ast::BinOp::Div,
                    lhs: lhs.type_of(),
                    rhs: rhs.type_of(),
                })
            }
        };
        Ok(output)
    }
}

impl Div for Value {
    type Output = OpResult;

    fn div(self, rhs: Self) -> Self::Output {
        Div::div(&self, &rhs)
    }
}

impl Rem for &Value {
    type Output = OpResult;

    fn rem(self, rhs: Self) -> Self::Output {
        let output = match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs % rhs),
            (lhs, rhs) => {
                return Err(ExceptionKind::UnsupportedOperandTypes {
                    op: ast::BinOp::Div,
                    lhs: lhs.type_of(),
                    rhs: rhs.type_of(),
                })
            }
        };
        Ok(output)
    }
}

impl Rem for Value {
    type Output = OpResult;

    fn rem(self, rhs: Self) -> Self::Output {
        Rem::rem(&self, &rhs)
    }
}

impl Not for &Value {
    type Output = OpResult;

    fn not(self) -> Self::Output {
        Ok(Value::Bool(!self.is_truthy()))
    }
}

impl Not for Value {
    type Output = OpResult;

    fn not(self) -> Self::Output {
        Not::not(&self)
    }
}

impl PartialEq for Value {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => lhs == rhs,
            (Value::Str(lhs), Value::Str(rhs)) => lhs == rhs,
            (Value::Bool(lhs), Value::Bool(rhs)) => lhs == rhs,
            (Value::Tuple(lhs_0, lhs_1), Value::Tuple(rhs_0, rhs_1)) => {
                lhs_0 == rhs_0 && lhs_1 == rhs_1
            }
            _ => false,
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Value::Int(lhs), Value::Int(rhs)) => Some(lhs.cmp(rhs)),
            (Value::Str(lhs), Value::Str(rhs)) => Some(lhs.cmp(rhs)),
            (Value::Bool(lhs), Value::Bool(rhs)) => Some(lhs.cmp(rhs)),
            (Value::Tuple(lhs_0, lhs_1), Value::Tuple(rhs_0, rhs_1)) => {
                let ord_0 = lhs_0.partial_cmp(rhs_0)?;
                if ord_0.is_eq() {
                    lhs_1.partial_cmp(rhs_1)
                } else {
                    Some(ord_0)
                }
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Str(s) => write!(f, "{s}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Tuple(fst, snd) => write!(f, "({fst}, {snd})"),
            Value::Closure(_) => write!(f, "<#closure>"),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Str => write!(f, "Str"),
            Type::Int => write!(f, "Int"),
            Type::Bool => write!(f, "Bool"),
            Type::Tuple => write!(f, "Tuple"),
            Type::Closure => write!(f, "Closure"),
        }
    }
}
