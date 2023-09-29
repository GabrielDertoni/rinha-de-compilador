use std::{borrow::Cow, collections::HashMap, rc::Rc};

use pretty::{docs, Doc, DocAllocator, RefDoc};

use ast::{from_json::BasicContext, AstNode, VisitContext, Visitor};

use crate::sem::{DefKind, RefKind, SemanticContext};

type DocBuilder<'a> = pretty::DocBuilder<'a, pretty::Arena<'a>>;
type BuildDoc<'a> = pretty::BuildDoc<'a, pretty::RefDoc<'a>, ()>;

/*
#[derive(Debug, Clone)]
pub enum CStmt {
    If(CIfStmt),
    Return(CReturnStmt),
    Expr(CExprStmt),
    Block(CBlockStmt),
}

#[derive(Debug, Clone)]
pub struct CIfStmt {
    pub condition: Box<CExpr>,
    pub then: Box<CStmt>,
    pub otherwise: Box<CStmt>,
}

#[derive(Debug, Clone)]
*/

const RESULT_VAR_NAME: &'static str = "result";
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
    // Variables to be hoisted up to the current function scope
    hoisted_vars: Vec<Cow<'static, str>>,
    curr: Option<ast::ExprId>,
}

impl<'a> LowerToC<'a> {
    pub fn new(sem: &'a SemanticContext) -> Self {
        LowerToC {
            code: None,
            sem,
            toplevel: BuildDoc::nil(),
            hoisted_vars: Vec::new(),
            curr: None,
        }
    }

    // Returns a "reference" to a hoisted variable
    pub fn new_var(&mut self, basename: impl Into<Cow<'static, str>>) -> usize {
        let id = self.hoisted_vars.len();
        self.hoisted_vars.push(basename.into());
        id
    }

    pub fn get_var(&mut self, alloc: &'a pretty::Arena<'a>, var: usize) -> DocBuilder<'a> {
        alloc.as_string(format_args!("{}_{var}", self.hoisted_vars[var]))
    }

    pub fn set(&mut self, code: DocBuilder<'a>) {
        debug_assert!(self.code.is_none(), "probably a bug");
        self.code = Some(code);
    }

    pub fn scope<R>(
        &mut self,
        alloc: &'a pretty::Arena<'a>,
        action: impl FnOnce(&mut Self) -> R,
    ) -> (R, DocBuilder<'a>) {
        debug_assert!(self.code.is_none(), "probably a bug");

        let save_hoisted = std::mem::take(&mut self.hoisted_vars);
        let result = action(self);
        let code = self.finish_scope(alloc);

        self.hoisted_vars = save_hoisted;
        (result, code)
    }

    pub fn append_toplevel(&mut self, doc: DocBuilder<'a>) {
        use pretty::Pretty;

        let alloc = doc.0;
        let toplevel = std::mem::take(&mut self.toplevel).pretty(alloc);
        self.toplevel = toplevel.append(doc).into();
    }

    // Returns the code for the expression that was built, the result of the evaluated
    // expression is going to be in the variable `result`
    pub fn finish_expr(&mut self) -> DocBuilder<'a> {
        self.code.take().expect("finishing without value")
    }

    // Finish the current expression and assign the `result` variable to another name. A fresh
    // variable will be created for storing the result and it will be hoisted.
    pub fn finish_expr_with_var(
        &mut self,
        basename: impl Into<Cow<'static, str>>,
    ) -> CExprResult<'a> {
        let code = self.finish_expr();
        let alloc = code.0;
        let var = self.new_var(basename);
        let var = self.get_var(alloc, var);

        CExprResult {
            code: code.append(assign_stmt(var.clone(), alloc.text(RESULT_VAR_NAME))),
            var,
        }
    }

    // Finish the current scope of the enclosing function
    pub fn finish_scope(&mut self, alloc: &'a pretty::Arena<'a>) -> DocBuilder<'a> {
        let hoisted = alloc.concat(
            std::mem::take(&mut self.hoisted_vars)
                .into_iter()
                .enumerate()
                .map(|(i, var)| {
                    let var = alloc.as_string(format_args!("{var}_{i}"));
                    stmt(docs![alloc, OBJECT_TYPE, Doc::space(), var])
                }),
        );
        // Include the `result` variable as well, which must always be present
        let hoisted =
            stmt(docs![alloc, OBJECT_TYPE, Doc::space(), RESULT_VAR_NAME]).append(hoisted);

        if let Some(code) = self.code.take() {
            hoisted.append(code)
        } else {
            hoisted
        }
    }

    // Return the completed C document
    pub fn finish(mut self, alloc: &'a pretty::Arena<'a>) -> DocBuilder<'a> {
        let scope = self.finish_scope(alloc);
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
        self.set(assign_result_to(code))
    }

    fn visit_fn_expr(&mut self, expr: &ast::FnExpr, cx: &Context<'a, Cx>) {
        use crate::sem::FreeVars;

        let mut free_vars = FreeVars::new();
        expr.accept(&mut free_vars, cx);
        let free_vars = free_vars.finish();

        // Declare the capture struct
        let n = expr.loc.0; // TODO: make this better
        let struct_name = cx.doc.as_string(format_args!("_closure_{n}"));
        self.append_toplevel(mk_struct(
            struct_name.clone(),
            // The first field must be a pointer to the function itself
            std::iter::once(
                cx.doc
                    .text("closure_t ")
                    .append(cx.doc.text(CLOSURE_FN_PTR_NAME)),
            )
            .chain(
                free_vars
                    .keys()
                    .map(|&var| mk_field(cx.doc, &cx.inner[var])),
            ),
        ));

        let fn_name = cx.doc.as_string(format_args!("_closure_fn_{n}"));
        let arg_list = cx.doc.text("arg_list");
        let params = [
            docs![cx.doc, struct_name.clone(), " ", CAPTURE_VAR_NAME],
            docs![cx.doc, OBJECT_TYPE, " ", arg_list.clone()],
        ];

        let result_var = cx.doc.text(RESULT_VAR_NAME);

        // Lets generate the function
        let fn_header = docs![
            cx.doc,
            OBJECT_TYPE,
            " ",
            fn_name.clone(),
            cx.doc.intersperse(params, cx.doc.text(",")).parens()
        ];

        let (_, fn_body) = self.scope(cx.doc, |vis| {
            // Inside the function, first load arguments from the argument list
            let load_args = cx
                .doc
                .concat(expr.params.iter().enumerate().flat_map(|(i, param)| {
                    // objec_t _user_var0;
                    // _user_var0 = get_arg_list(result, 0);
                    let var = user_var(cx.doc, &cx.inner[param.ident]);
                    [
                        decl_stmt(var.clone()),
                        assign_stmt(
                            var,
                            builtin_call2(get_arg_list, arg_list.clone(), cx.doc.as_string(i)),
                        ),
                    ]
                }));

            // Now finally generate the code for the function
            expr.body.accept(vis, cx);
            let code = vis.finish_expr();
            vis.set(docs![
                cx.doc,
                load_args,
                code,
                // return result;
                stmt(docs![cx.doc, "return", " ", result_var.clone()]),
            ])
        });

        let fn_def = fn_header.append(block(fn_body));
        self.append_toplevel(fn_def);

        let closure_var = cx.doc.as_string(format_args!("capture_{n}"));

        // Now load everything we need into the closure's capture struct
        let create_capture = docs![
            cx.doc,
            // _closure_N capture_n;
            stmt(docs![cx.doc, ptr(struct_name.clone()), Doc::space(), closure_var.clone()]),
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
            assign_result_to(closure_var),
        ];

        self.set(create_capture);
    }

    fn visit_if_expr(&mut self, expr: &ast::IfExpr, cx: &Context<'a, Cx>) {
        expr.condition.accept(self, cx);
        let condition = self.finish_expr_with_var("condition");

        expr.then.accept(self, cx);
        let then = self.finish_expr();

        expr.otherwise.accept(self, cx);
        let otherwise = self.finish_expr();

        self.set(docs![
            cx.doc,
            condition.code,
            cx.doc.reflow("if ").append(condition.var.parens()),
            block(then),
            cx.doc.reflow(" else "),
            block(otherwise),
            cx.doc.line(),
        ]);
    }

    fn visit_let_expr(&mut self, expr: &ast::LetExpr, cx: &Context<'a, Cx>) {
        expr.init.accept(self, cx);
        let init_code = self.finish_expr();
        let var_name = &cx.inner[expr.name.ident];

        expr.next.accept(self, cx);
        let in_code = self.finish_expr();

        let var = user_var(cx.doc, var_name);
        let result_var = cx.doc.text(RESULT_VAR_NAME);
        self.set(docs![
            cx.doc,
            init_code,
            // object_t _var_VAR;
            stmt(docs![cx.doc, OBJECT_TYPE, Doc::space(), var.clone()]),
            // _user_VAR = result;
            assign_stmt(var, result_var),
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

        expr.lhs.accept(self, cx);
        let lhs = self.finish_expr_with_var("lhs");

        expr.rhs.accept(self, cx);
        let rhs = self.finish_expr_with_var("rhs");

        self.set(lhs.merge(rhs, |lhs, rhs| {
            assign_result_to(
                cx.doc
                    .intersperse([lhs, cx.doc.text(op), rhs], cx.doc.space()),
            )
        }));
    }

    fn visit_lit_expr(&mut self, expr: &ast::LitExpr, cx: &Context<'a, Cx>) {
        let doc = match expr {
            ast::LitExpr::Str(lit) => assign_result_to(builtin_call1(
                mk_static_str,
                cx.doc.as_string(&lit.value).double_quotes(),
            )),
            ast::LitExpr::Int(int) => {
                assign_result_to(builtin_call1(mk_int, cx.doc.as_string(int.value)))
            }
            ast::LitExpr::Bool(bool) => {
                assign_result_to(builtin_call1(mk_bool, cx.doc.as_string(bool.value)))
            }
            ast::LitExpr::Tuple(tup) => {
                tup.first.accept(self, cx);
                let fst = self.finish_expr_with_var("fst");

                tup.second.accept(self, cx);
                let snd = self.finish_expr_with_var("snd");

                fst.merge(snd, |fst, snd| {
                    assign_result_to(builtin_call2(mk_tuple, fst, snd))
                })
            }
        };
        self.set(doc);
    }

    fn visit_call_expr(&mut self, expr: &ast::CallExpr, cx: &Context<'a, Cx>) {
        expr.callee.accept(self, cx);
        let callee = self.finish_expr_with_var("callee");

        let mut code = callee.code;
        let mut args = Vec::new();

        for arg in &expr.args {
            arg.accept(self, cx);
            let arg = self.finish_expr_with_var("arg");
            code = code.append(arg.code);
            args.push(arg.var);
        }

        let result_var = cx.doc.text(RESULT_VAR_NAME);

        self.set(docs![
            cx.doc,
            code,
            // result = mk_arg_list(N);
            assign_result_to(builtin_call1(
                mk_arg_list,
                cx.doc.as_string(expr.args.len())
            )),
            // set_arg_list(result, 0, arg0);
            // set_arg_list(result, 1, arg1);
            // ...and so on
            cx.doc
                .concat(args.iter().enumerate().map(|(i, arg)| stmt(builtin_call3(
                    set_arg_list,
                    result_var.clone(),
                    cx.doc.as_string(i),
                    arg.clone()
                )))),
            // result = call(callee, result /* the arg list */);
            assign_result_to(builtin_call2(call, callee.var, result_var)),
        ]);
    }

    fn visit_builtin_expr(&mut self, expr: &ast::BuiltinExpr, cx: &Context<'a, Cx>) {
        let mut code: DocBuilder<'a> = cx.doc.nil();
        let mut vars = Vec::new();

        for arg in &expr.args {
            arg.accept(self, cx);
            let result = self.finish_expr_with_var("arg");

            code = code.append(result.code);
            vars.push(result.var);
        }

        let fn_name = cx.doc.as_string(&cx.inner[expr.name]).double_quotes();

        let result_var = cx.doc.text(RESULT_VAR_NAME);

        self.set(docs![
            cx.doc,
            code,
            // result = mk_arg_list(N);
            assign_result_to(builtin_call1(
                mk_arg_list,
                cx.doc.as_string(expr.args.len())
            )),
            // set_arg_list(result, 0, arg0);
            // set_arg_list(result, 1, arg1);
            // ...and so on
            cx.doc
                .concat(vars.iter().enumerate().map(|(i, arg)| stmt(builtin_call3(
                    set_arg_list,
                    result_var.clone(),
                    cx.doc.as_string(i),
                    arg.clone()
                )))),
            // result = extern_call("builtin_name", result);
            assign_result_to(builtin_call2(extern_call, fn_name, result_var)),
        ]);
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

    mk_arg_list/1;
    set_arg_list/3;
    get_arg_list/2;
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

fn assign_result_to<'a>(value: DocBuilder<'a>) -> DocBuilder<'a> {
    let alloc = value.0;
    assign_stmt(alloc.text(RESULT_VAR_NAME), value)
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
        let result = vis.finish_expr();

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
