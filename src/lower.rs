use std::rc::Rc;

use pretty::{docs, Doc, DocAllocator, Pretty};

use ast::from_json::BasicContext;

use crate::sem::SemanticContext;

type BuildDoc<'a> = pretty::BuildDoc<'a, pretty::RefDoc<'a>, ()>;
type DocBuilder<'a> = pretty::DocBuilder<'a, DocAlloc<'a>, ()>;

const CAPTURE_VAR_NAME: &'static str = "capture";
const CLOSURE_FN_PTR_NAME: &'static str = "base";
const OBJECT_TYPE: &'static str = "object_t";

macro_rules! fmt {
    ($alloc:ident, $($args:tt)*) => {
        $alloc.as_string(format_args!($($args)*))
    };
}

#[derive(Debug, Clone)]
pub struct LowerToC<'a> {
    cx: Rc<BasicContext>,
    sem: Rc<SemanticContext>,

    // Code that is goingo the top of the C output
    toplevel: BuildDoc<'a>,
    // How many C variables we need to allocate hoisted
    hoisted_vars: usize,
    // How many C variables are live
    live_vars: usize,
}

impl<'a> LowerToC<'a> {
    pub fn new(cx: Rc<BasicContext>, sem: Rc<SemanticContext>) -> Self {
        LowerToC {
            cx,
            sem,
            toplevel: BuildDoc::nil(),
            hoisted_vars: 1,
            live_vars: 1,
        }
    }

    pub fn tmp_var(&self) -> Place {
        Place::Register(self.live_vars)
    }

    /// Mark the last temporary variable as "used" so that it won't be clobbered by other operations inside
    /// of the scope.
    pub fn push_var(&mut self) {
        self.live_vars += 1;
        self.hoisted_vars = self.hoisted_vars.max(self.live_vars);
    }

    /// Evaluate an expression and save the result to a fresh variable. `action` should build
    /// an expression inside of `self` that will be finished once it returns.
    pub fn eval(
        &mut self,
        doc: &'a DocAlloc<'a>,
        action: impl FnOnce(&mut Self, Place) -> DocBuilder<'a>,
    ) -> CExprResult<'a> {
        // The result of the expression will be stored in this variable
        let var = self.tmp_var();
        let code = action(self, var.clone());

        // Mark the value as used. We only need to do this here, since `action` may
        // use `var` as it wishes, so long as it writes the final value to it in
        // the end of the expression. However, now we have the value we wanted in
        // the variable and should no longer change it within the scope.
        self.push_var();

        CExprResult { code, var }
    }

    pub fn eval_c(
        mut self,
        doc: &'a DocAlloc<'a>,
        action: impl FnOnce(&mut Self, Place) -> DocBuilder<'a>,
    ) -> DocBuilder<'a> {
        let var = self.tmp_var();
        let code = action(&mut self, var);
        self.finish(doc, code)
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
        doc: &'a DocAlloc<'a>,
        action: impl FnOnce(&mut Self, Place) -> R,
    ) -> R {
        let save_hoisted = self.hoisted_vars;
        let save_live = self.live_vars;

        self.hoisted_vars = 0;
        self.live_vars = 0;
        let var = self.tmp_var();
        let result = action(self, var);

        self.hoisted_vars = save_hoisted;
        self.live_vars = save_live;
        result
    }

    pub fn append_toplevel(&mut self, doc: DocBuilder<'a>) {
        let alloc = doc.0;
        let toplevel = std::mem::take(&mut self.toplevel).pretty(alloc);
        self.toplevel = toplevel.append(alloc.hardline()).append(doc).into();
    }

    // Finish the current scope of the enclosing function
    pub fn gen_hoisted(&mut self, doc: &'a DocAlloc<'a>) -> DocBuilder<'a> {
        doc.stmts((0..self.hoisted_vars).map(|i| doc.obj_decl_stmt(fmt!(doc, "_{i}"))))
    }

    // Return the completed C document
    pub fn finish(mut self, doc: &'a DocAlloc<'a>, main: DocBuilder<'a>) -> DocBuilder<'a> {
        let scope = self.gen_hoisted(doc);
        doc.stmts([
            doc.text("#include \"rt.h\""),
            self.toplevel.pretty(doc),
            doc.wrap_in_main(scope.append(doc.hardline()).append(main)),
        ])
    }

    fn load_var(&self, ident: ast::Ident, doc: &'a DocAlloc<'a>) -> Place {
        Place::var(&self.cx[ident])
        /*
        let id = self.curr.expect("inside expr");
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
    // NOTE: In practice, will always be a `Place::Register`
    pub var: Place,
}

impl<'a> CExprResult<'a> {
    pub fn merge(
        self,
        other: Self,
        combine: impl FnOnce(Place, Place) -> DocBuilder<'a>,
    ) -> DocBuilder<'a> {
        let doc = self.code.0;
        self.code
            .append(doc.hardline())
            .append(other.code)
            .append(doc.hardline())
            .append(combine(self.var, other.var))
    }
}

#[derive(Debug, Clone)]
pub enum Place {
    Register(usize),
    Var(Rc<str>),
}

impl Place {
    fn var(name: impl AsRef<str>) -> Self {
        Place::Var(name.as_ref().into())
    }
}

impl<'a, D, A> Pretty<'a, D, A> for Place
where
    A: 'a,
    D: ?Sized + DocAllocator<'a, A>,
{
    fn pretty(self, allocator: &'a D) -> pretty::DocBuilder<'a, D, A> {
        match self {
            Place::Register(id) => fmt!(allocator, "_{id}"),
            Place::Var(name) => fmt!(allocator, "_user_{name}"),
        }
    }
}

impl<'a> LowerToC<'a> {
    pub fn gen_expr_id(
        &mut self,
        expr: ast::ExprId,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
        let cx = self.cx.clone();
        self.gen_expr(&cx[expr], result_var, doc)
    }

    pub fn gen_expr(
        &mut self,
        expr: &ast::Expr,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
        match expr {
            ast::Expr::Fn(expr) => self.gen_fn_expr(expr, result_var, doc),
            ast::Expr::If(expr) => self.gen_if_expr(expr, result_var, doc),
            ast::Expr::Let(expr) => self.gen_let_expr(expr, result_var, doc),
            ast::Expr::Bin(expr) => self.gen_bin_expr(expr, result_var, doc),
            ast::Expr::Lit(expr) => self.gen_lit_expr(expr, result_var, doc),
            ast::Expr::Var(expr) => self.gen_var_expr(expr, result_var, doc),
            ast::Expr::Call(expr) => self.gen_call_expr(expr, result_var, doc),
            ast::Expr::Builtin(expr) => self.gen_builtin_expr(expr, result_var, doc),
        }
    }

    pub fn gen_var_expr(
        &mut self,
        expr: &ast::VarExpr,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
        let var = self.load_var(expr.ident, doc);
        doc.assign_place(result_var, var)
    }

    pub fn gen_fn_expr(
        &mut self,
        expr: &ast::FnExpr,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
        use crate::sem::FreeVars;

        let free_vars = FreeVars::analyze(expr, &*self.cx);

        // Declare the capture struct
        let n = expr.loc.0; // TODO: make this better
        let struct_name = fmt!(doc, "_capture_{n}");
        self.append_toplevel(
            doc.c_struct(
                struct_name.clone(),
                free_vars
                    .keys()
                    .map(|&var| doc.field(OBJECT_TYPE, &self.cx[var])),
            ),
        );

        let fn_name = fmt!(doc, "_closure_fn_{n}");
        let arg_list = doc.text("arg_list");
        let params = [
            doc.param(doc.ptr(struct_name.clone()), CAPTURE_VAR_NAME),
            doc.param(OBJECT_TYPE, arg_list.clone()),
        ];

        // Lets generate the function

        let fn_body = self.fn_scope(doc, |vis, result_var| {
            // Inside the function, first load arguments from the argument list
            let load_args = doc.stmts(expr.params.iter().enumerate().flat_map(|(i, param)| {
                // objec_t _user_var0;
                // _user_var0 = get_arg(result, 0);
                let var = Place::var(&vis.cx[param.ident]);
                [
                    doc.obj_decl_stmt(var.clone()),
                    doc.assign_place(
                        var,
                        doc.builtin_call2(get_arg, arg_list.clone(), doc.as_string(i)),
                    ),
                ]
            }));

            // Now finally generate the code for the function
            let body = vis.gen_expr_id(expr.body, result_var.clone(), doc);
            doc.stmts([
                load_args,
                body,
                // return result;
                doc.return_stmt(result_var),
            ])
        });

        let fn_def = doc.c_fn(OBJECT_TYPE, fn_name.clone(), params, fn_body);
        self.append_toplevel(fn_def);

        let closure_var = fmt!(doc, "capture_{n}");

        // Now load everything we need into the closure's capture struct
        doc.stmts([
            // _closure_N capture_n;
            doc.decl_stmt(doc.ptr(struct_name.clone()), closure_var.clone(), None),
            // closure = (_closure_N*)malloc(sizeof(CLOSURE_STRUCT));
            doc.assign_stmt(
                closure_var.clone(),
                doc.cast(
                    doc.ptr(struct_name.clone()),
                    doc.call_expr(doc.text("malloc"), [doc.sizeof(struct_name)]),
                ),
            ),
            // closure->base = (void*)_closure_fn_0;
            doc.assign_stmt(
                doc.ptr_field(closure_var.clone(), CLOSURE_FN_PTR_NAME),
                doc.cast(doc.text("void*"), fn_name.clone()),
            ),
            // closure->_user_var0 = _user_var0;
            // closure->_user_var1 = _user_var1;
            // ...and so on
            doc.stmts(free_vars.keys().map(|&ident| {
                let var = doc.user_var(&self.cx[ident]);
                doc.assign_stmt(
                    doc.ptr_field(closure_var.clone(), var),
                    self.load_var(ident, doc),
                )
            })),
            // result = closure;
            doc.assign_place(result_var, closure_var),
        ])
    }

    pub fn gen_if_expr(
        &mut self,
        expr: &ast::IfExpr,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
        let var = self.tmp_var();
        let condition = self.gen_expr_id(expr.condition, var.clone(), doc);
        let then = self.gen_expr_id(expr.then, result_var.clone(), doc);
        let otherwise = self.gen_expr_id(expr.otherwise, result_var, doc);

        doc.stmts([condition, doc.if_stmt(var, then, otherwise)])
    }

    pub fn gen_let_expr(
        &mut self,
        expr: &ast::LetExpr,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
        let var_name = &self.cx[expr.name.ident];
        let var = Place::var(var_name);

        doc.stmts([
            // object_t _var_VAR = mk_var_uninit();
            doc.decl_stmt(
                OBJECT_TYPE,
                var.clone(),
                Some(doc.builtin_call0(mk_var_uninit)),
            ),
            self.gen_expr_id(expr.init, var, doc),
            self.gen_expr_id(expr.next, result_var, doc),
        ])
    }

    pub fn gen_bin_expr(
        &mut self,
        expr: &ast::BinExpr,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
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
            let lhs = vis.eval(doc, |vis, var| vis.gen_expr_id(expr.lhs, var, doc));
            let rhs = vis.eval(doc, |vis, var| vis.gen_expr_id(expr.rhs, var, doc));

            lhs.merge(rhs, |lhs, rhs| {
                doc.assign_place(
                    result_var,
                    doc.intersperse(
                        [lhs.pretty(doc), doc.text(op), rhs.pretty(doc)],
                        doc.space(),
                    ),
                )
            })
        })
    }

    pub fn gen_lit_expr(
        &mut self,
        expr: &ast::LitExpr,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
        match expr {
            ast::LitExpr::Str(lit) => doc.assign_place(
                result_var,
                doc.builtin_call1(mk_static_str, doc.as_string(&lit.value).double_quotes()),
            ),
            ast::LitExpr::Int(int) => doc.assign_place(
                result_var,
                doc.builtin_call1(mk_int, doc.as_string(int.value)),
            ),
            ast::LitExpr::Bool(bool) => doc.assign_place(
                result_var,
                doc.builtin_call1(mk_bool, doc.as_string(bool.value)),
            ),
            ast::LitExpr::Tuple(tup) => self.scope(|vis| {
                let fst = vis.eval(doc, |vis, var| vis.gen_expr_id(tup.first, var, doc));
                let snd = vis.eval(doc, |vis, var| vis.gen_expr_id(tup.second, var, doc));

                fst.merge(snd, |fst, snd| {
                    doc.assign_place(
                        result_var,
                        doc.builtin_call2(mk_tuple, fst.pretty(doc), snd.pretty(doc)),
                    )
                })
            }),
        }
    }

    pub fn gen_call_expr(
        &mut self,
        expr: &ast::CallExpr,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
        self.scope(|vis| {
            let callee = vis.eval(doc, |vis, var| vis.gen_expr_id(expr.callee, var, doc));

            let mut code = callee.code;
            let mut args = Vec::new();

            for &arg in &expr.args {
                let arg = vis.eval(doc, |vis, var| vis.gen_expr_id(arg, var, doc));
                code = code.append(doc.hardline()).append(arg.code);
                args.push(arg.var);
            }

            let tmp = vis.tmp_var();

            doc.stmts([
                code,
                // tmp = mk_args(N);
                doc.assign_place(
                    tmp.clone(),
                    doc.builtin_call1(mk_args, doc.as_string(expr.args.len())),
                ),
                // set_arg(tmp, 0, arg0);
                // set_arg(tmp, 1, arg1);
                // ...and so on
                doc.stmts(args.into_iter().enumerate().map(|(i, arg)| {
                    doc.stmt(doc.builtin_call3(
                        set_arg,
                        tmp.clone().pretty(doc),
                        doc.as_string(i),
                        arg.pretty(doc),
                    ))
                })),
                // result = call(callee, result /* the arg list */);
                doc.assign_place(result_var, doc.builtin_call2(call, callee.var.pretty(doc), tmp.pretty(doc))),
            ])
        })
    }

    pub fn gen_builtin_expr(
        &mut self,
        expr: &ast::BuiltinExpr,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
        self.scope(|vis| {
            let mut code: DocBuilder<'a> = doc.nil();
            let mut vars = Vec::new();

            for &arg in &expr.args {
                let result = vis.eval(doc, |vis, var| vis.gen_expr_id(arg, var, doc));

                code = code.append(result.code);
                vars.push(result.var);
            }

            let fn_name = doc.as_string(&vis.cx[expr.name]).double_quotes();
            let args_var = vis.tmp_var();

            doc.stmts([
                code,
                // result = mk_args(N);
                doc.assign_place(
                    args_var.clone(),
                    doc.builtin_call1(mk_args, doc.as_string(expr.args.len())),
                ),
                // set_arg(result, 0, arg0);
                // set_arg(result, 1, arg1);
                // ...and so on
                doc.stmts(vars.into_iter().enumerate().map(|(i, arg)| {
                    doc.stmt(doc.builtin_call3(set_arg, args_var.clone().pretty(doc), doc.as_string(i), arg.pretty(doc)))
                })),
                // result = extern_call("builtin_name", result);
                doc.assign_place(
                    result_var,
                    doc.builtin_call2(extern_call, fn_name, args_var.pretty(doc)),
                ),
            ])
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

    mk_var_uninit/0;
    var_assign/2;

    call/2;
    extern_call/2;

    mk_args/1;
    set_arg/3;
    get_arg/2;

    // Operations
    add/2;
    sub/2;
    mul/2;
    div/2;
    eq/2;
    neq/2;
    le/2;
    leq/2;
    ge/2;
    geq/2;
}

pub struct DocAlloc<'a> {
    arena: pretty::Arena<'a>,
}

impl<'a> DocAlloc<'a> {
    pub fn new() -> Self {
        DocAlloc {
            arena: pretty::Arena::new(),
        }
    }
}

impl<'a> DocAllocator<'a, ()> for DocAlloc<'a> {
    type Doc = pretty::RefDoc<'a>;

    fn alloc(&'a self, doc: Doc<'a, Self::Doc>) -> Self::Doc {
        self.arena.alloc(doc)
    }

    fn alloc_column_fn(
        &'a self,
        f: impl Fn(usize) -> Self::Doc + 'a,
    ) -> <Self::Doc as pretty::DocPtr<'a, ()>>::ColumnFn {
        self.arena.alloc_column_fn(f)
    }

    fn alloc_width_fn(
        &'a self,
        f: impl Fn(isize) -> Self::Doc + 'a,
    ) -> <Self::Doc as pretty::DocPtr<'a, ()>>::WidthFn {
        self.arena.alloc_width_fn(f)
    }
}

impl<'a> DocAlloc<'a> {
    fn builtin_call0<F>(&'a self, _marker: F) -> DocBuilder<'a>
    where
        F: Builtin + BuiltinArity<0>,
    {
        self.text(F::NAME).append(self.nil().parens())
    }

    fn builtin_call1<F>(&'a self, _marker: F, arg: DocBuilder<'a>) -> DocBuilder<'a>
    where
        F: Builtin + BuiltinArity<1>,
    {
        self.text(F::NAME).append(arg.parens())
    }

    fn builtin_call2<F>(
        &'a self,
        _marker: F,
        arg1: DocBuilder<'a>,
        arg2: DocBuilder<'a>,
    ) -> DocBuilder<'a>
    where
        F: Builtin + BuiltinArity<2>,
    {
        self.call_expr(self.text(F::NAME), [arg1, arg2])
    }

    fn builtin_call3<F>(
        &'a self,
        _marker: F,
        arg1: DocBuilder<'a>,
        arg2: DocBuilder<'a>,
        arg3: DocBuilder<'a>,
    ) -> DocBuilder<'a>
    where
        F: Builtin + BuiltinArity<3>,
    {
        self.call_expr(self.text(F::NAME), [arg1, arg2, arg3])
    }

    fn call_expr(
        &'a self,
        func: DocBuilder<'a>,
        args: impl IntoIterator<Item = DocBuilder<'a>>,
    ) -> DocBuilder<'a> {
        docs![
            self,
            func,
            self.block_indent(self.intersperse(args, self.concat([self.text(","), self.line()])))
                .group()
                .parens()
        ]
    }

    fn ptr(&'a self, ty: DocBuilder<'a>) -> DocBuilder<'a> {
        ty.append(self.text("*"))
    }

    fn cast(&'a self, target_ty: DocBuilder<'a>, value: DocBuilder<'a>) -> DocBuilder<'a> {
        target_ty.parens().append(value)
    }

    fn sizeof(&'a self, ty: DocBuilder<'a>) -> DocBuilder<'a> {
        self.call_expr(self.text("sizeof"), [ty])
    }

    fn assign_stmt(
        &'a self,
        var: impl Pretty<'a, Self>,
        value: impl Pretty<'a, Self>,
    ) -> DocBuilder<'a> {
        self.stmt(docs![self, var, self.reflow(" = "), value])
    }

    fn assign_place(&'a self, place: Place, value: impl Pretty<'a, Self>) -> DocBuilder<'a> {
        match &place {
            Place::Register(_) => self.assign_stmt(place, value),
            Place::Var(_) => self.builtin_call2(var_assign, place.pretty(self), value.pretty(self)),
        }
    }

    fn ptr_field(
        &'a self,
        ptr: impl Pretty<'a, Self>,
        field: impl Pretty<'a, Self>,
    ) -> DocBuilder<'a> {
        docs![self, ptr, "->", field]
    }

    fn if_stmt(
        &'a self,
        condition: impl Pretty<'a, Self>,
        then: impl Pretty<'a, Self>,
        otherwise: impl Pretty<'a, Self>,
    ) -> DocBuilder<'a> {
        docs![
            self,
            "if ",
            self.block_indent(condition.pretty(self)).group().parens(),
            self.softline(),
            self.block(then.pretty(self)),
            " else ",
            self.block(otherwise.pretty(self))
        ]
    }

    fn obj_decl_stmt(&'a self, var: impl Pretty<'a, Self>) -> DocBuilder<'a> {
        self.decl_stmt(OBJECT_TYPE, var, Some(self.text("NULL")))
    }

    fn decl_stmt(
        &'a self,
        ty: impl Pretty<'a, Self>,
        var: impl Pretty<'a, Self>,
        init: Option<DocBuilder<'a>>,
    ) -> DocBuilder<'a> {
        let init = if let Some(init) = init {
            docs![self, " = ", init]
        } else {
            self.nil()
        };
        self.stmt(docs![self, ty, self.space(), var, init])
    }

    fn stmt(&'a self, doc: DocBuilder<'a>) -> DocBuilder<'a> {
        doc.append(self.text(";"))
    }

    fn return_stmt(&'a self, value: impl Pretty<'a, Self>) -> DocBuilder<'a> {
        docs![self, "return ", value]
    }

    fn block(&'a self, doc: DocBuilder<'a>) -> DocBuilder<'a> {
        self.block_indent(doc).braces()
        /*use pretty::block::*;

        BlockDoc {
            affixes: vec![Affixes::new(self.text("{"), self.text("}")), Affixes::new(self.line(), self.line())],
            body: doc,
        }.format(4)*/
    }

    fn param(&'a self, ty: impl Pretty<'a, Self>, name: impl Pretty<'a, Self>) -> DocBuilder<'a> {
        docs![self, ty, self.space(), name]
    }

    fn c_fn_header<I>(
        &'a self,
        ret_type: impl Pretty<'a, Self>,
        name: impl Pretty<'a, Self>,
        params: I,
    ) -> DocBuilder<'a>
    where
        I: IntoIterator,
        I::IntoIter: Clone,
        I::Item: Pretty<'a, Self>,
    {
        let iter = params.into_iter();
        let softcomma = self.concat([self.text(","), self.softline()]);
        let comma = self.concat([self.text(","), self.line()]);
        let params_aligned = self.intersperse(iter.clone(), softcomma).align();
        let params = params_aligned.union(self.block_indent(self.intersperse(iter, comma)).group());
        docs![self, ret_type, self.softline(), name, params.parens()]
    }

    fn c_fn<I>(
        &'a self,
        ret_type: impl Pretty<'a, Self>,
        name: impl Pretty<'a, Self>,
        params: I,
        body: impl Pretty<'a, Self>,
    ) -> DocBuilder<'a>
    where
        I: IntoIterator,
        I::IntoIter: Clone,
        I::Item: Pretty<'a, Self>,
    {
        let header = self.c_fn_header(ret_type, name, params);
        docs![self, header, self.softline(), self.block(body.pretty(self))]
    }

    fn c_struct(
        &'a self,
        name: DocBuilder<'a>,
        fields: impl IntoIterator<Item = DocBuilder<'a>>,
    ) -> DocBuilder<'a> {
        self.stmt(docs![
            self,
            "typedef struct ",
            self.block(self.concat(fields.into_iter().map(|f| self.stmt(f)))),
            self.space(),
            name,
        ])
    }

    fn field(&'a self, ty: &'a str, name: &str) -> DocBuilder<'a> {
        docs![self, ty, " ", self.user_var(name)]
    }

    fn user_var(&'a self, name: &str) -> DocBuilder<'a> {
        self.as_string(format_args!("_user_{name}"))
    }

    fn wrap_in_main(&'a self, body: DocBuilder<'a>) -> DocBuilder<'a> {
        self.c_fn(
            "int",
            "main",
            [self.text("void")],
            body.append(self.line()).append(self.return_stmt("0")),
        )
    }

    fn comment(&'a self, text: &'a str) -> DocBuilder<'a> {
        self.text("// ").append(text)
    }

    fn block_indent(&'a self, inside: impl Pretty<'a, Self>) -> DocBuilder<'a> {
        docs![self, self.line_().append(inside).nest(4), self.line_()]
    }

    fn stmts<I>(&'a self, ls: I) -> DocBuilder<'a>
    where
        I: IntoIterator,
        I::Item: Pretty<'a, Self>,
    {
        self.intersperse(ls, self.hardline())
    }

    fn lines<I>(&'a self, ls: I) -> DocBuilder<'a>
    where
        I: IntoIterator,
        I::Item: Pretty<'a, Self>,
    {
        self.intersperse(ls, self.line())
    }
}

/*
impl<Cx: VisitContext> VisitContext for Context<'_, Cx> {}

impl<Cx: VisitContext> std::ops::Index<ast::ExprId> for Context<'_, Cx> {
    type Output = ast::Expr;

    fn index(&self, index: ast::ExprId) -> &Self::Output {
        &self.inner[index]
    }
}
 */

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
