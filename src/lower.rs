use pretty::{docs, Doc, DocAllocator};

use ast::{AstNode, VisitContext, Visitor};

use crate::sem::SemanticContext;

type DocBuilder<'a> = pretty::DocBuilder<'a, pretty::Arena<'a>>;
type BuildDoc<'a> = pretty::BuildDoc<'a, pretty::RefDoc<'a>, ()>;

const CAPTURE_VAR_NAME: &'static str = "capture";
const CLOSURE_FN_PTR_NAME: &'static str = "base";
const OBJECT_TYPE: &'static str = "object_t";

#[derive(Debug, Clone)]
pub struct LowerToC<'a> {
    // Code for last visited expression
    code: Option<DocBuilder<'a>>,
    sem: &'a SemanticContext,

    // Code that is goingo the top of the C output
    toplevel: BuildDoc<'a>,
    // How many C variables we need to allocate hoisted
    hoisted_vars: usize,
    // How many C variables are live
    live_vars: usize,
    // Where to write the result of the current expression
    result_var: Option<DocBuilder<'a>>,
    curr: Option<ast::ExprId>,
}

impl<'a> LowerToC<'a> {
    pub fn new(sem: &'a SemanticContext) -> Self {
        LowerToC {
            code: None,
            sem,
            toplevel: BuildDoc::nil(),
            hoisted_vars: 1,
            live_vars: 1,
            result_var: None,
            curr: None,
        }
    }

    // Returns the name of a hoisted variable, allocating a new one if necessary
    pub fn alloc_var(&mut self, alloc: &'a pretty::Arena<'a>) -> DocBuilder<'a> {
        let id = self.live_vars;
        self.live_vars += 1;
        self.hoisted_vars = self.hoisted_vars.max(self.live_vars);
        alloc.as_string(format_args!("_{id}"))
    }

    pub fn next_var(&self, alloc: &'a pretty::Arena<'a>) -> DocBuilder<'a> {
        let id = self.live_vars;
        alloc.as_string(format_args!("_{id}"))
    }

    pub fn push_var(&mut self) {
        self.live_vars += 1;
        self.hoisted_vars = self.hoisted_vars.max(self.live_vars);
    }

    pub fn set(&mut self, code: DocBuilder<'a>) {
        debug_assert!(self.code.is_none(), "probably a bug");
        self.code = Some(code);
    }

    pub fn result_var(&self, alloc: &'a pretty::Arena<'a>) -> DocBuilder<'a> {
        self.result_var
            .clone()
            .expect("should have some variable to put the output")
    }

    /// Evaluate an expression and write the result to `var`. `action` should `set` to an
    /// expression which will be finished when it returns.
    pub fn eval_to(
        &mut self,
        var: DocBuilder<'a>,
        action: impl FnOnce(&mut Self),
    ) -> DocBuilder<'a> {
        let save_result_var = self.result_var.replace(var);
        action(self);
        let code = self.finish_expr().code;
        self.result_var = save_result_var;
        code
    }

    /// Evaluate an expression and save the result to a fresh variable. `action` should build
    /// an expression inside of `self` that will be finished once it returns.
    pub fn eval(
        &mut self,
        alloc: &'a pretty::Arena<'a>,
        action: impl FnOnce(&mut Self),
    ) -> CExprResult<'a> {
        // The result of the expression will be stored in this variable
        let var = self.next_var(alloc);
        let save_result_var = self.result_var.replace(var);
        action(self);
        let result = self.finish_expr();
        self.result_var = save_result_var;

        // Mark the value as used. We only need to do this here, since `action` may
        // use `var` as it wishes, so long as it writes the final value to it in
        // the end of the expression. However, now we have the value we wanted in
        // the variable and should no longer change it within the scope.
        self.push_var();

        result
    }

    pub fn eval_c(
        mut self,
        alloc: &'a pretty::Arena<'a>,
        action: impl FnOnce(&mut Self),
    ) -> DocBuilder<'a> {
        let result = self.next_var(alloc);
        self.result_var = Some(result);
        action(&mut self);
        self.result_var = None;
        self.finish(alloc)
    }

    // A scope allows for use of variables local to it, it also calls `finish_expr` at the end.
    // The result of evaluating the expression will be stored on a new variable. `action`
    // should generate code for one expression.
    pub fn scope<R>(&mut self, action: impl FnOnce(&mut Self) -> R) -> R {
        let save_live_vars = self.live_vars;
        let result = action(self);
        self.live_vars = save_live_vars;
        result
    }

    pub fn fn_scope<R>(
        &mut self,
        alloc: &'a pretty::Arena<'a>,
        action: impl FnOnce(&mut Self) -> R,
    ) -> (R, DocBuilder<'a>) {
        debug_assert!(self.code.is_none(), "probably a bug");

        let save_hoisted = self.hoisted_vars;
        let save_live = self.live_vars;

        self.hoisted_vars = 0;
        self.live_vars = 0;
        let result = action(self);
        let code = self.finish_fn(alloc);

        self.hoisted_vars = save_hoisted;
        self.live_vars = save_live;
        (result, code)
    }

    pub fn append_toplevel(&mut self, doc: DocBuilder<'a>) {
        use pretty::Pretty;

        let alloc = doc.0;
        let toplevel = std::mem::take(&mut self.toplevel).pretty(alloc);
        self.toplevel = toplevel.append(doc).into();
    }

    fn assign_result_to(&self, value: DocBuilder<'a>) -> DocBuilder<'a> {
        let alloc = value.0;
        assign_stmt(self.result_var(alloc), value)
    }

    // Returns the code for the expression that was built, the result of the evaluated
    // expression is going to be in the variable `result`
    pub fn finish_expr(&mut self) -> CExprResult<'a> {
        let code = self.code.take().expect("finishing without value");
        let var = self.result_var(code.0);
        CExprResult { code, var }
    }

    // Finish the current scope of the enclosing function
    pub fn finish_fn(&mut self, alloc: &'a pretty::Arena<'a>) -> DocBuilder<'a> {
        let hoisted = alloc.concat((0..self.hoisted_vars).map(|i| {
            let var = alloc.as_string(format_args!("_{i}"));
            stmt(docs![alloc, OBJECT_TYPE, Doc::space(), var])
        }));

        if let Some(code) = self.code.take() {
            hoisted.append(code)
        } else {
            hoisted
        }
    }

    // Return the completed C document
    pub fn finish(mut self, alloc: &'a pretty::Arena<'a>) -> DocBuilder<'a> {
        let scope = self.finish_fn(alloc);
        docs![
            alloc,
            "#include \"rt.h\"",
            alloc.line(),
            self.toplevel,
            alloc.line(),
            wrap_in_main(scope),
        ]
    }

    fn load_var<Cx>(&self, ident: ast::Ident, cx: &Context<'a, Cx>) -> DocBuilder<'a>
    where
        Cx: VisitContext,
        Cx: std::ops::Index<ast::Ident, Output = str>,
    {
        let id = self.curr.expect("inside expr");

        let var = user_var(cx.doc, &cx.inner[ident]);
        var
        /*
        let info = self.sem.var_info(id);
        match info.ref_kind {
            RefKind::Local | RefKind::Param { .. } => var,
            RefKind::Captured { .. } => docs![cx.doc, CAPTURE_VAR_NAME, "->", var],
        }
        */
    }
}

#[derive(Debug, Clone)]
pub struct CExprResult<'a> {
    pub code: DocBuilder<'a>,
    pub var: DocBuilder<'a>,
}

impl<'a> CExprResult<'a> {
    pub fn nil(alloc: &'a pretty::Arena<'a>) -> Self {
        CExprResult {
            code: alloc.nil(),
            var: alloc.nil(),
        }
    }

    pub fn merge(
        self,
        other: Self,
        combine: impl FnOnce(DocBuilder<'a>, DocBuilder<'a>) -> DocBuilder<'a>,
    ) -> DocBuilder<'a> {
        let alloc = self.code.0;
        self.code
            .append(other.code)
            .append(combine(self.var, other.var))
    }
}

pub struct Context<'a, Cx> {
    pub inner: Cx,
    pub doc: &'a pretty::Arena<'a>,
}

impl<'a, Cx> Visitor<Context<'a, Cx>> for LowerToC<'a>
where
    Cx: VisitContext,
    Cx: std::ops::Index<ast::Ident, Output = str>,
{
    fn visit_expr_id(&mut self, expr: &ast::ExprId, cx: &Context<'a, Cx>) {
        self.curr.replace(*expr);
        expr.visit_children(self, cx);
    }

    fn visit_expr(&mut self, expr: &ast::Expr, cx: &Context<'a, Cx>) {
        expr.visit_children(self, cx);
    }

    fn visit_var_expr(&mut self, expr: &ast::VarExpr, cx: &Context<'a, Cx>) {
        let code = self.load_var(expr.ident, cx);
        self.set(self.assign_result_to(code))
    }

    fn visit_fn_expr(&mut self, expr: &ast::FnExpr, cx: &Context<'a, Cx>) {
        use crate::sem::FreeVars;

        let mut free_vars = FreeVars::new();
        expr.accept(&mut free_vars, cx);
        let free_vars = free_vars.finish();

        // Declare the capture struct
        let n = expr.loc.0; // TODO: make this better
        let struct_name = cx.doc.as_string(format_args!("_capture_{n}"));
        self.append_toplevel(mk_struct(
            struct_name.clone(),
            free_vars
                .keys()
                .map(|&var| mk_field(cx.doc, &cx.inner[var])),
        ));

        let fn_name = cx.doc.as_string(format_args!("_closure_fn_{n}"));
        let arg_list = cx.doc.text("arg_list");
        let params = [
            docs![cx.doc, ptr(struct_name.clone()), " ", CAPTURE_VAR_NAME],
            docs![cx.doc, OBJECT_TYPE, " ", arg_list.clone()],
        ];

        // Lets generate the function
        let fn_header = docs![
            cx.doc,
            OBJECT_TYPE,
            " ",
            fn_name.clone(),
            cx.doc
                .intersperse(params, cx.doc.reflow(", "))
                .align()
                .parens()
        ];

        let (_, fn_body) = self.fn_scope(cx.doc, |vis| {
            // Inside the function, first load arguments from the argument list
            let load_args = cx
                .doc
                .concat(expr.params.iter().enumerate().flat_map(|(i, param)| {
                    // objec_t _user_var0;
                    // _user_var0 = get_arg(result, 0);
                    let var = user_var(cx.doc, &cx.inner[param.ident]);
                    [
                        decl_stmt(var.clone()),
                        assign_stmt(
                            var,
                            builtin_call2(get_arg, arg_list.clone(), cx.doc.as_string(i)),
                        ),
                    ]
                }));

            // Now finally generate the code for the function
            let result = vis.eval(cx.doc, |vis| expr.body.accept(vis, cx));
            vis.set(docs![
                cx.doc,
                load_args,
                result.code,
                // return result;
                stmt(docs![cx.doc, "return", " ", result.var]),
            ])
        });

        let fn_def = fn_header.append(cx.doc.space()).append(block(fn_body));
        self.append_toplevel(fn_def);

        let closure_var = cx.doc.as_string(format_args!("capture_{n}"));

        // Now load everything we need into the closure's capture struct
        let create_capture = docs![
            cx.doc,
            // _closure_N capture_n;
            stmt(docs![
                cx.doc,
                ptr(struct_name.clone()),
                Doc::space(),
                closure_var.clone()
            ]),
            // closure = (_closure_N)malloc(sizeof(CLOSURE_STRUCT));
            assign_stmt(
                closure_var.clone(),
                ptr(struct_name.clone()).parens().append(call_expr(
                    cx.doc.text("malloc"),
                    [call_expr(cx.doc.text("sizeof"), [struct_name])]
                ))
            ),
            // closure->base = (void*)_closure_fn_0;
            assign_stmt(
                docs![cx.doc, closure_var.clone(), "->", CLOSURE_FN_PTR_NAME],
                docs![cx.doc, "(void*)", fn_name.clone()]
            ),
            // closure->_user_var0 = _user_var0;
            // closure->_user_var1 = _user_var1;
            // ...and so on
            cx.doc.concat(free_vars.keys().map(|&ident| {
                let var = user_var(cx.doc, &cx.inner[ident]);
                assign_stmt(
                    docs![cx.doc, closure_var.clone(), "->", var],
                    self.load_var(ident, cx),
                )
            })),
            // result = closure;
            self.assign_result_to(closure_var),
        ];

        self.set(create_capture);
    }

    fn visit_if_expr(&mut self, expr: &ast::IfExpr, cx: &Context<'a, Cx>) {
        let condition =
            self.scope(|vis| vis.eval(cx.doc, |vis| expr.condition.accept(vis, cx)));

        expr.then.accept(self, cx);
        let then = self.finish_expr().code;

        expr.otherwise.accept(self, cx);
        let otherwise = self.finish_expr().code;

        self.set(docs![
            cx.doc,
            condition.code,
            cx.doc.reflow("if ").append(condition.var.parens()),
            " ",
            block(then),
            cx.doc.reflow(" else "),
            block(otherwise),
            cx.doc.line(),
        ]);
    }

    fn visit_let_expr(&mut self, expr: &ast::LetExpr, cx: &Context<'a, Cx>) {
        let var_name = &cx.inner[expr.name.ident];
        let var = user_var(cx.doc, var_name);

        let init_code = self.eval_to(var.clone(), |vis| expr.init.accept(vis, cx));

        expr.next.accept(self, cx);
        let in_code = self.finish_expr().code;

        self.set(docs![
            cx.doc,
            // object_t _var_VAR;
            stmt(docs![cx.doc, OBJECT_TYPE, Doc::space(), var.clone()]),
            init_code,
            in_code,
        ])
    }

    fn visit_bin_expr(&mut self, expr: &ast::BinExpr, cx: &Context<'a, Cx>) {
        let op: &'static str = match expr.op {
            ast::BinOp::Add => "+",
            ast::BinOp::Sub => "-",
            ast::BinOp::Mul => "*",
            ast::BinOp::Div => "/",
            ast::BinOp::Rem => "%",
            ast::BinOp::Eq => "==",
            ast::BinOp::Neq => "!=",
            ast::BinOp::Lt => "<",
            ast::BinOp::Gt => ">",
            ast::BinOp::Lte => "<=",
            ast::BinOp::Gte => ">=",
            ast::BinOp::And => "&&",
            ast::BinOp::Or => "||",
        };

        self.scope(|vis| {
            let lhs = vis.eval(cx.doc, |vis| expr.lhs.accept(vis, cx));
            let rhs = vis.eval(cx.doc, |vis| expr.rhs.accept(vis, cx));

            vis.set(lhs.merge(rhs, |lhs, rhs| {
                vis.assign_result_to(
                    cx.doc
                        .intersperse([lhs, cx.doc.text(op), rhs], cx.doc.space()),
                )
            }));
        });
    }

    fn visit_lit_expr(&mut self, expr: &ast::LitExpr, cx: &Context<'a, Cx>) {
        let doc = match expr {
            ast::LitExpr::Str(lit) => self.assign_result_to(builtin_call1(
                mk_static_str,
                cx.doc.as_string(&lit.value).double_quotes(),
            )),
            ast::LitExpr::Int(int) => {
                self.assign_result_to(builtin_call1(mk_int, cx.doc.as_string(int.value)))
            }
            ast::LitExpr::Bool(bool) => {
                self.assign_result_to(builtin_call1(mk_bool, cx.doc.as_string(bool.value)))
            }
            ast::LitExpr::Tuple(tup) => self.scope(|vis| {
                let fst = vis.eval(cx.doc, |vis| tup.first.accept(vis, cx));
                let snd = vis.eval(cx.doc, |vis| tup.second.accept(vis, cx));

                fst.merge(snd, |fst, snd| {
                    vis.assign_result_to(builtin_call2(mk_tuple, fst, snd))
                })
            }),
        };
        self.set(doc);
    }

    fn visit_call_expr(&mut self, expr: &ast::CallExpr, cx: &Context<'a, Cx>) {
        self.scope(|vis| {
            let callee = vis.eval(cx.doc, |vis| expr.callee.accept(vis, cx));

            let mut code = callee.code;
            let mut args = Vec::new();

            for arg in &expr.args {
                let arg = vis.eval(cx.doc, |vis| arg.accept(vis, cx));
                code = code.append(arg.code);
                args.push(arg.var);
            }

            vis.set(docs![
                cx.doc,
                code,
                // result = mk_args(N);
                vis.assign_result_to(builtin_call1(mk_args, cx.doc.as_string(expr.args.len()))),
                // set_arg(result, 0, arg0);
                // set_arg(result, 1, arg1);
                // ...and so on
                cx.doc
                    .concat(args.iter().enumerate().map(|(i, arg)| stmt(builtin_call3(
                        set_arg,
                        vis.result_var(cx.doc),
                        cx.doc.as_string(i),
                        arg.clone()
                    )))),
                // result = call(callee, result /* the arg list */);
                vis.assign_result_to(builtin_call2(call, callee.var, vis.result_var(cx.doc))),
            ]);
        });
    }

    fn visit_builtin_expr(&mut self, expr: &ast::BuiltinExpr, cx: &Context<'a, Cx>) {
        self.scope(|vis| {
            let mut code: DocBuilder<'a> = cx.doc.nil();
            let mut vars = Vec::new();

            for arg in &expr.args {
                let result = vis.eval(cx.doc, |vis| arg.accept(vis, cx));

                code = code.append(result.code);
                vars.push(result.var);
            }

            let fn_name = cx.doc.as_string(&cx.inner[expr.name]).double_quotes();
            let args_var = vis.alloc_var(cx.doc);

            vis.set(docs![
                cx.doc,
                code,
                // result = mk_args(N);
                assign_stmt(args_var, builtin_call1(mk_args, cx.doc.as_string(expr.args.len()))),
                // set_arg(result, 0, arg0);
                // set_arg(result, 1, arg1);
                // ...and so on
                cx.doc
                    .concat(vars.iter().enumerate().map(|(i, arg)| stmt(builtin_call3(
                        set_arg,
                        vis.result_var(cx.doc),
                        cx.doc.as_string(i),
                        arg.clone()
                    )))),
                // result = extern_call("builtin_name", result);
                vis.assign_result_to(builtin_call2(extern_call, fn_name, vis.result_var(cx.doc))),
            ]);
        })
    }
}

macro_rules! declare_builtins {
    ($($name:ident / $arity:literal;)*) => {
        $(
            #[allow(non_camel_case_types)]
            pub struct $name;

            impl Builtin for $name {
                const ARITY: usize = $arity;
                const NAME: &'static str = concat!("_rt__builtin__", stringify!($name));
            }

            impl BuiltinArity<$arity> for $name {}
        )*
    };
}

pub trait Builtin {
    const ARITY: usize;
    const NAME: &'static str;
}

pub trait BuiltinArity<const N: usize> {}

declare_builtins! {
    // Constructors of primitive types
    mk_static_str/1;
    mk_int/1;
    mk_bool/1;
    mk_tuple/2;

    call/2;
    extern_call/2;

    mk_args/1;
    set_arg/3;
    get_arg/2;
}

fn builtin_call1<'a, F>(_marker: F, arg: DocBuilder<'a>) -> DocBuilder<'a>
where
    F: Builtin + BuiltinArity<1>,
{
    let alloc = arg.0;
    alloc.text(F::NAME).append(arg.parens())
}

fn builtin_call2<'a, F>(_marker: F, arg1: DocBuilder<'a>, arg2: DocBuilder<'a>) -> DocBuilder<'a>
where
    F: Builtin + BuiltinArity<2>,
{
    let alloc = arg1.0;
    call_expr(alloc.text(F::NAME), [arg1, arg2])
}

fn builtin_call3<'a, F>(
    _marker: F,
    arg1: DocBuilder<'a>,
    arg2: DocBuilder<'a>,
    arg3: DocBuilder<'a>,
) -> DocBuilder<'a>
where
    F: Builtin + BuiltinArity<3>,
{
    let alloc = arg1.0;
    call_expr(alloc.text(F::NAME), [arg1, arg2, arg3])
}

fn call_expr<'a>(
    func: DocBuilder<'a>,
    args: impl IntoIterator<Item = DocBuilder<'a>>,
) -> DocBuilder<'a> {
    let alloc = func.0;
    docs![
        alloc,
        func,
        alloc.intersperse(args, alloc.reflow(", ")).parens()
    ]
}

fn ptr<'a>(ty: DocBuilder<'a>) -> DocBuilder<'a> {
    let alloc = ty.0;
    ty.append(alloc.text("*"))
}

fn assign_stmt<'a>(var: DocBuilder<'a>, value: DocBuilder<'a>) -> DocBuilder<'a> {
    let alloc = var.0;
    stmt(docs![alloc, var, alloc.reflow(" = "), value])
}

fn decl_stmt<'a>(var: DocBuilder<'a>) -> DocBuilder<'a> {
    let alloc = var.0;
    stmt(docs![alloc, OBJECT_TYPE, " ", var])
}

fn stmt<'a>(doc: DocBuilder<'a>) -> DocBuilder<'a> {
    let alloc = doc.0;
    doc.append(alloc.text(";")).append(alloc.hardline())
}

fn block<'a>(doc: DocBuilder<'a>) -> DocBuilder<'a> {
    let alloc = doc.0;
    doc.indent(4).enclose(alloc.line(), alloc.nil()).braces()
}

fn mk_field<'a>(alloc: &'a pretty::Arena<'a>, name: &str) -> DocBuilder<'a> {
    docs![alloc, alloc.text(OBJECT_TYPE), " ", user_var(alloc, name)]
}

fn mk_struct<'a>(
    name: DocBuilder<'a>,
    fields: impl IntoIterator<Item = DocBuilder<'a>>,
) -> DocBuilder<'a> {
    let alloc = name.0;

    stmt(docs![
        alloc,
        alloc.text("typedef struct "),
        block(alloc.concat(fields.into_iter().map(stmt))),
        alloc.space(),
        name,
    ])
}

fn user_var<'a>(alloc: &'a pretty::Arena<'a>, name: &str) -> DocBuilder<'a> {
    alloc.as_string(format_args!("_user_{name}"))
}

fn wrap_in_main<'a>(doc: DocBuilder<'a>) -> DocBuilder<'a> {
    let alloc = doc.0;
    let header = alloc.reflow("int main()");
    let footer = stmt(alloc.reflow("return 0"));
    header.append(block(doc.append(footer)))
}

impl<Cx: VisitContext> VisitContext for Context<'_, Cx> {}

impl<Cx: VisitContext> std::ops::Index<ast::ExprId> for Context<'_, Cx> {
    type Output = ast::Expr;

    fn index(&self, index: ast::ExprId) -> &Self::Output {
        &self.inner[index]
    }
}

#[cfg(test)]
mod test {
    use std::{fmt::Write, rc::Rc};

    use ast::from_json::BasicContext;

    use super::*;

    fn to_c<Node: AstNode>(parse_cx: Rc<BasicContext>, tree: Node) -> String {
        let doc_alloc = pretty::Arena::new();
        let cx = Context {
            inner: parse_cx.clone(),
            doc: &doc_alloc,
        };
        let sem = SemanticContext::new(parse_cx);
        let mut vis = LowerToC::new(&sem);

        tree.accept(&mut vis, &cx);
        let result = vis.finish_expr().code;

        let mut out = String::new();
        result.render_fmt(80, &mut out).unwrap();

        out
    }

    fn dummy_location() -> serde_json::Value {
        serde_json::json!({
            "kind": "Location",
            "start": 0,
            "end": 0,
            "filename": "test.rinha"
        })
    }

    #[test]
    pub fn test_bin() {
        let mut parse_cx = BasicContext::new();

        let loc = dummy_location();
        let tree: ast::BinExpr = ast::from_json!(parse_cx, {
            "kind": "Binary",
            "op": "Add",
            "lhs": {
                "kind": "Str",
                "value": "hello, world",
                "location": loc.clone()
            },
            "rhs": {
                "kind": "Int",
                "value": 10,
                "location": loc.clone()
            },
            "location": loc.clone()
        })
        .unwrap();

        let c = to_c(Rc::new(parse_cx), tree);
        println!("{c}");
    }

    #[test]
    pub fn test_builtin() {
        let mut parse_cx = BasicContext::new();

        let loc = dummy_location();
        let tree: ast::BuiltinExpr = ast::from_json!(parse_cx, {
            "kind": "Print",
            "value": {
                "kind": "Str",
                "value": "hello, world",
                "location": loc.clone(),
            },
            "location": loc.clone()
        })
        .unwrap();

        let c = to_c(Rc::new(parse_cx), tree);
        println!("{c}");
    }
}
