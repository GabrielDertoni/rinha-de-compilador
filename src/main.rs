mod ast;
mod exception;
mod from_json;
mod value;

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use clap::Parser;

use exception::*;
use from_json::Interner;
use value::*;

#[derive(Debug, Parser)]
#[command(name = "rinha-compiler-rust")]
struct Args {
    file: PathBuf,
}

pub type Env = HashMap<ast::Ident, Value>;

pub type EvalResult = Result<Value, Exception>;

pub type Builtin = fn(&[Value], &Context) -> EvalResult;

#[derive(Clone)]
pub struct Context<'caller> {
    cx: Rc<from_json::BasicContext>,
    builtins: Rc<HashMap<ast::Ident, Builtin>>,
    env: Env,
    frame: Frame,
    parent: Option<&'caller Context<'caller>>,
}

impl Context<'static> {
    pub fn with_builtins(
        cx: from_json::BasicContext,
        builtins: HashMap<ast::Ident, Builtin>,
        frame: Frame,
    ) -> Self {
        Context {
            cx: Rc::new(cx),
            builtins: Rc::new(builtins),
            env: HashMap::new(),
            frame,
            parent: None,
        }
    }
}

impl<'caller> Context<'caller> {
    pub fn extend_mut(&mut self, name: ast::Ident, value: Value) {
        self.env.insert(name, value);
    }

    pub fn try_extend(&self, name: ast::Ident, value: Value) -> Option<Context<'caller>> {
        use std::collections::hash_map::Entry;

        let mut cx = self.clone();
        if let Entry::Vacant(entry) = cx.env.entry(name) {
            entry.insert(value);
            Some(cx)
        } else {
            None
        }
    }

    pub fn lookup(&self, name: ast::Ident) -> Option<&Value> {
        self.env.get(&name)
    }

    pub fn lookup_builtin(&self, name: ast::Ident) -> Option<Builtin> {
        self.builtins.get(&name).copied()
    }

    pub fn traceback(&self) -> Vec<Frame> {
        let mut frames = Vec::new();
        let mut curr = Some(self);
        while let Some(cx) = curr {
            frames.push(cx.frame.clone());
            curr = cx.parent;
        }
        frames
    }

    pub fn eval(&self, expr: ast::ExprId) -> EvalResult {
        match &self.cx[expr] {
            ast::Expr::Fn(func) => Ok(Value::Closure(Rc::new(Closure {
                capture: self.env.clone(),
                func: func.clone(),
                named: None,
            }))),
            ast::Expr::If(expr) => {
                if self.eval(expr.condition)?.is_truthy() {
                    self.eval(expr.then)
                } else {
                    self.eval(expr.otherwise)
                }
            }
            ast::Expr::Let(expr) => {
                let value = if let ast::Expr::Fn(fn_expr) = &self.cx[expr.init] {
                    let closure = Closure {
                        capture: self.env.clone(),
                        func: fn_expr.clone(),
                        named: Some(expr.name.ident.clone()),
                    };
                    Value::Closure(Rc::new(closure))
                } else {
                    self.eval(expr.init)?
                };

                if let Some(cx) = self.try_extend(expr.name.ident.clone(), value) {
                    cx.eval(expr.next)
                } else {
                    Err(ExceptionKind::VariableAlreadyDefined {
                        name: expr.name.ident.clone(),
                    }
                    .within(self))
                }
            }
            &ast::Expr::Bin(ast::BinExpr { op, lhs, rhs, .. }) => {
                // Handle the lazy ones first
                match op {
                    ast::BinOp::And => {
                        let lhs = self.eval(lhs)?;
                        return if lhs.is_truthy() {
                            self.eval(rhs)
                        } else {
                            Ok(lhs)
                        };
                    }
                    ast::BinOp::Or => {
                        let lhs = self.eval(lhs)?;
                        return if lhs.is_truthy() {
                            Ok(lhs)
                        } else {
                            self.eval(rhs)
                        };
                    }
                    _ => (),
                }

                let lhs = self.eval(lhs)?;
                let rhs = self.eval(rhs)?;
                match op {
                    ast::BinOp::Add => (lhs + rhs).map_err(|e| e.within(self)),
                    ast::BinOp::Sub => (lhs - rhs).map_err(|e| e.within(self)),
                    ast::BinOp::Mul => (lhs * rhs).map_err(|e| e.within(self)),
                    ast::BinOp::Div => (lhs / rhs).map_err(|e| e.within(self)),
                    ast::BinOp::Rem => (lhs % rhs).map_err(|e| e.within(self)),
                    ast::BinOp::Eq => Ok(Value::Bool(lhs == rhs)),
                    ast::BinOp::Neq => Ok(Value::Bool(lhs != rhs)),
                    ast::BinOp::Lt => Ok(Value::Bool(lhs < rhs)),
                    ast::BinOp::Gt => Ok(Value::Bool(lhs > rhs)),
                    ast::BinOp::Lte => Ok(Value::Bool(lhs <= rhs)),
                    ast::BinOp::Gte => Ok(Value::Bool(lhs >= rhs)),
                    ast::BinOp::And | ast::BinOp::Or => unreachable!("already handled"),
                }
            }
            &ast::Expr::Var(ast::VarExpr { ident, .. }) => {
                if let Some(value) = self.lookup(ident) {
                    Ok(value.clone())
                } else {
                    Err(ExceptionKind::VariableNotDefined {
                        name: ident.clone(),
                    }
                    .within(self))
                }
            }
            ast::Expr::Call(expr) => {
                let callee = self.eval(expr.callee)?;
                if let Value::Closure(closure) = &callee {
                    if closure.func.params.len() != expr.args.len() {
                        let missing = closure
                            .func
                            .params
                            .iter()
                            .skip(expr.args.len())
                            .map(|param| self.cx[param.ident].to_owned())
                            .collect();

                        return Err(ExceptionKind::WrongNumberOfArgs {
                            arity: closure.func.params.len(),
                            actual: expr.args.len(),
                            missing,
                        }
                        .within(self));
                    }

                    let mut closure_env = closure.capture.clone();
                    if let Some(name) = &closure.named {
                        closure_env.insert(name.clone(), Value::Closure(Rc::clone(&closure)));
                    }

                    let mut cx = Context {
                        cx: self.cx.clone(),
                        builtins: self.builtins.clone(),
                        env: closure_env,
                        frame: Frame {
                            caller: expr.loc.clone(),
                            fn_name: closure.named.clone(),
                            fn_loc: closure.func.loc.clone(),
                        },
                        parent: Some(self),
                    };
                    for (param, &arg) in closure.func.params.iter().zip(&expr.args) {
                        cx.extend_mut(param.ident.clone(), self.eval(arg)?);
                    }
                    cx.eval(closure.func.body)
                } else {
                    return Err(ExceptionKind::ObjectNotCallable {
                        ty: callee.type_of(),
                    }
                    .within(self));
                }
            }
            ast::Expr::Builtin(expr) => {
                let Some(func) = self.lookup_builtin(expr.name) else {
                    return Err(ExceptionKind::VariableNotDefined { name: expr.name }.within(self));
                };

                let values = expr
                    .args
                    .iter()
                    .map(|&arg| self.eval(arg))
                    .collect::<Result<Vec<_>, _>>()?;

                func(&values, self)
            }
            ast::Expr::Lit(lit) => match lit {
                ast::LitExpr::Str(s) => Ok(Value::Str(s.value.clone())),
                ast::LitExpr::Bool(b) => Ok(Value::Bool(b.value)),
                ast::LitExpr::Int(i) => Ok(Value::Int(i.value)),
                ast::LitExpr::Tuple(tup) => Ok(Value::Tuple(
                    Box::new(self.eval(tup.first)?),
                    Box::new(self.eval(tup.second)?),
                )),
            },
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let content = std::fs::read_to_string(&args.file)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let mut cx = from_json::BasicContext::new();
    let ast: ast::File = from_json::parse(&json, &mut cx)?;

    let mut builtins = HashMap::new();
    // builtins.insert(cx.intern_static("first"), builtin_first as Builtin);
    // builtins.insert(cx.intern_static("second"), builtin_second as Builtin);
    builtins.insert(cx.intern_static("Print"), builtin_print as Builtin);

    let frame = Frame {
        caller: ast.loc.clone(),
        fn_name: None,
        fn_loc: ast.loc.clone(),
    };

    // for e in &cx.exprs {
    //     dbg!(&e);
    // }

    let cx = Context::with_builtins(cx, builtins, frame);
    match cx.eval(ast.expr) {
        Ok(value) => println!(">> {value}"),
        Err(error) => eprintln!("{error}"),
    }

    Ok(())
}

fn assert_args(expected: &[&str], n_args: usize) -> Result<(), ExceptionKind> {
    if expected.len() == n_args {
        Ok(())
    } else {
        let missing = expected
            .iter()
            .skip(n_args)
            .map(|&param| param.to_owned())
            .collect();

        Err(ExceptionKind::WrongNumberOfArgs {
            arity: expected.len(),
            actual: n_args,
            missing,
        })
    }
}

fn builtin_print(args: &[Value], cx: &Context) -> EvalResult {
    assert_args(&["value"], args.len()).map_err(|e| e.within(cx))?;
    println!("{}", args[0]);
    Ok(args[0].clone())
}

/*
fn builtin_first(args: &[Value], cx: &Context) -> EvalResult {
    assert_args(&["tuple"], args.len()).map_err(|e| e.within(cx))?;
    let Value::Tuple(fst, _snd) = &args[0] else {
        return Err(ExceptionKind::WrongArgumentType {
            param: ast::Param {
                ident: String::from("tuple"),
                loc: ast::loc::builtin(),
            },
            expected: Type::Tuple,
            got: args[0].type_of(),
        }
        .within(cx));
    };
    Ok((**fst).clone())
}

fn builtin_second(args: &[Value], cx: &Context) -> EvalResult {
    assert_args(&["tuple"], args.len()).map_err(|e| e.within(cx))?;
    let Value::Tuple(_fst, snd) = &args[0] else {
        return Err(ExceptionKind::WrongArgumentType {
            param: ast::Param {
                ident: String::from("tuple"),
                loc: ast::loc::builtin(),
            },
            expected: Type::Tuple,
            got: args[0].type_of(),
        }
        .within(cx));
    };
    Ok((**snd).clone())
}
*/
