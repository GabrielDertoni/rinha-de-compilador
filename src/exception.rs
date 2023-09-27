use crate::ast;
use crate::{Context, Type};

#[derive(Debug, Clone)]
pub enum ExceptionKind {
    // NameError
    VariableNotDefined {
        name: ast::Ident,
    },
    VariableAlreadyDefined {
        name: ast::Ident,
    },

    // TypeError
    UnsupportedOperandTypes {
        op: ast::BinOp,
        lhs: Type,
        rhs: Type,
    },
    WrongArgumentType {
        param: ast::Param,
        expected: Type,
        got: Type,
    },
    ObjectNotCallable {
        ty: Type,
    },
    WrongNumberOfArgs {
        arity: usize,
        actual: usize,
        missing: Vec<String>,
    },
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub caller: ast::Location,
    pub fn_name: Option<ast::Ident>,
    pub fn_location: ast::Location,
}

#[derive(Debug, Clone)]
pub struct Exception {
    pub kind: ExceptionKind,
    pub traceback: Vec<Frame>,
}

impl ExceptionKind {
    pub fn within(self, cx: &Context) -> Exception {
        Exception {
            kind: self,
            traceback: cx.traceback(),
        }
    }
}

impl std::fmt::Display for Exception {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Error:")?;
        writeln!(f, "    Traceback:")?;
        for frame in &self.traceback {
            writeln!(f, "      {frame}")?;
        }
        writeln!(f, "{}", self.kind)
    }
}

impl std::fmt::Display for ExceptionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // NameError
            ExceptionKind::VariableNotDefined { name } => {
                write!(f, "NameError: name '{name}' is not defined")
            }
            ExceptionKind::VariableAlreadyDefined { name } => {
                write!(f, "NameError: name '{name}' is already defined")
            }

            // TypeError
            ExceptionKind::UnsupportedOperandTypes { op, lhs, rhs } => write!(
                f,
                "TypeError: unsupported operand types for {op}: {lhs} and {rhs}"
            ),
            ExceptionKind::WrongArgumentType {
                param,
                expected,
                got,
            } => write!(
                f,
                "TypeError: wrong argument type for '{param}': expected {expected} but got {got}",
                param = param.ident
            ),
            ExceptionKind::ObjectNotCallable { ty } => {
                write!(f, "TypeError: object of type {ty} is not callable")
            }
            ExceptionKind::WrongNumberOfArgs {
                arity,
                actual,
                missing,
            } => {
                if missing.len() > 0 {
                    write!(f, "TypeError: not enough arguments, missing ")?;
                    for (i, miss) in missing.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{miss}")?;
                    }
                    Ok(())
                } else {
                    write!(
                        f,
                        "TypeError: too many arguments, expected {arity} but got {actual}"
                    )
                }
            }
        }
    }
}

impl std::fmt::Display for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(fn_name) = &self.fn_name {
            write!(
                f,
                "File '{}', at {} in function {}",
                self.caller.file, self.caller.start, fn_name
            )
        } else {
            write!(
                f,
                "File {}, at {} in function <#closure:{}:{}>",
                self.caller.file, self.caller.start, self.fn_location.file, self.fn_location.start
            )
        }
    }
}
