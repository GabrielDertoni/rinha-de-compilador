use std::rc::Rc;

use pretty::{docs, Doc, DocAllocator, Pretty};

use ast::from_json::BasicContext;

use crate::sem::SemanticContext;

type BuildDoc<'a> = pretty::BuildDoc<'a, pretty::RefDoc<'a>, ()>;
type DocBuilder<'a> = pretty::DocBuilder<'a, DocAlloc<'a>, ()>;

const CAPTURE_VAR_NAME: &'static str = "capture";
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
    allocd_registers: usize,
    // How many C variables are live
    live_regs: usize,
}

impl<'a> LowerToC<'a> {
    pub fn new(cx: Rc<BasicContext>, sem: Rc<SemanticContext>) -> Self {
        LowerToC {
            cx,
            sem,
            toplevel: BuildDoc::nil(),
            allocd_registers: 0,
            live_regs: 0,
        }
    }

    // Returns the place for a temporary variable. The variable will have type `object_t`.
    pub fn tmp_var(&self) -> Place {
        Place::Register(self.live_regs)
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
    pub fn scope<R>(&mut self, vars: usize, action: impl FnOnce(&mut Self, &[Place]) -> R) -> R {
        let save_live_vars = self.live_regs;

        // Allocate the registers
        let regs = (0..vars)
            .map(|i| self.live_regs + i)
            .map(Place::Register)
            .collect::<Vec<_>>();
        self.live_regs += vars;
        self.allocd_registers = self.allocd_registers.max(self.live_regs);

        let result = action(self, &regs);
        self.live_regs = save_live_vars;
        result
    }

    pub fn fn_scope<R>(&mut self, action: impl FnOnce(&mut Self, Place) -> R) -> R {
        let save_hoisted = self.allocd_registers;
        let save_live = self.live_regs;

        self.allocd_registers = 0;
        self.live_regs = 0;
        let var = self.tmp_var();
        let result = action(self, var.into());

        self.allocd_registers = save_hoisted;
        self.live_regs = save_live;
        result
    }

    pub fn append_toplevel(&mut self, doc: DocBuilder<'a>) {
        let alloc = doc.0;
        let toplevel = std::mem::take(&mut self.toplevel).pretty(alloc);
        self.toplevel = toplevel.append(alloc.hardline()).append(doc).into();
    }

    // Finish the current scope of the enclosing function
    pub fn gen_register_allocs(&mut self, doc: &'a DocAlloc<'a>) -> DocBuilder<'a> {
        doc.stmts(
            // There is always at least one register allocated. This way code can always assume
            // it can use a tmp variable from `self.tmp_var()`.
            (0..=self.allocd_registers)
                .map(Place::Register)
                .map(|reg| doc.obj_decl_stmt(reg.pretty(doc))),
        )
    }

    // Return the completed C document
    pub fn finish(mut self, doc: &'a DocAlloc<'a>, main: DocBuilder<'a>) -> DocBuilder<'a> {
        let scope = self.gen_register_allocs(doc);
        doc.stmts([
            doc.text("#include \"rt.h\""),
            self.toplevel.pretty(doc),
            doc.wrap_in_main(scope.append(doc.hardline()).append(main)),
        ])
    }

    fn load_var(&self, ident: ast::Ident) -> Place {
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

#[derive(Debug, Clone, Copy)]
pub struct Register(pub usize);

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
        let var = self.load_var(expr.ident);
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

        let fn_body = self.fn_scope(|vis, result_var| {
            // Inside the function, first load arguments from the argument list
            let load_args = doc.stmts(expr.params.iter().enumerate().map(|(i, param)| {
                // objec_t _user_var0;
                // _user_var0 = get_arg(result, 0);
                let var = Place::var(&vis.cx[param.ident]);
                doc.decl_stmt(
                    OBJECT_TYPE,
                    var.clone(),
                    Some(doc.builtin_call2(get_arg, arg_list.clone(), doc.as_string(i))),
                )
            }));

            let load_captures =
                doc.stmts(free_vars.keys().map(|&var| {
                    let var = doc.user_var(&vis.cx[var]);
                    doc.decl_stmt(
                        OBJECT_TYPE,
                        var.clone(),
                        Some(doc.assign_stmt(
                            var.clone(),
                            doc.ptr_field(doc.text(CAPTURE_VAR_NAME), var),
                        )),
                    )
                }));

            // Now finally generate the code for the function
            let body = vis.gen_expr_id(expr.body, result_var.clone(), doc);
            doc.stmts([
                vis.gen_register_allocs(doc),
                load_captures,
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
            // _capture_0* capture_0 = (_capture_0*)mk_closure(N_CAPTURED, (void*)_closure_fn_0);
            doc.decl_stmt(
                doc.ptr(struct_name.clone()),
                closure_var.clone(),
                Some(doc.cast(
                    doc.ptr(struct_name.clone()),
                    doc.builtin_call2(
                        mk_closure,
                        doc.as_string(free_vars.len()),
                        doc.cast(doc.text("void*"), fn_name.clone()),
                    ),
                )),
            ),
            // capture_0->_user_var0 = _user_var0;
            // capture_0->_user_var1 = _user_var1;
            // ...and so on
            doc.stmts(free_vars.keys().map(|&ident| {
                let var = doc.user_var(&self.cx[ident]);
                doc.assign_stmt(
                    doc.ptr_field(closure_var.clone(), var),
                    self.load_var(ident),
                )
            })),
            // result = closure_obj(capture_0);
            doc.assign_place(
                result_var,
                doc.builtin_call1(
                    closure_obj,
                    doc.cast(doc.ptr(doc.text(OBJECT_TYPE)), closure_var),
                ),
            ),
        ])
    }

    pub fn gen_if_expr(
        &mut self,
        expr: &ast::IfExpr,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
        let condition_var = self.tmp_var();
        let condition_code = self.gen_expr_id(expr.condition, condition_var.clone(), doc);
        let then = self.gen_expr_id(expr.then, result_var.clone(), doc);
        let otherwise = self.gen_expr_id(expr.otherwise, result_var, doc);

        doc.stmts([
            condition_code,
            doc.if_stmt(
                doc.builtin_call1(read_bool, condition_var.pretty(doc)),
                then,
                otherwise,
            ),
        ])
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
        let op_builtin: &'static str = match expr.op {
            ast::BinOp::Add => add::NAME,
            ast::BinOp::Sub => sub::NAME,
            ast::BinOp::Mul => mul::NAME,
            ast::BinOp::Div => div::NAME,
            ast::BinOp::Rem => rem::NAME,
            ast::BinOp::Eq => eq::NAME,
            ast::BinOp::Neq => neq::NAME,
            ast::BinOp::Lt => lt::NAME,
            ast::BinOp::Gt => gt::NAME,
            ast::BinOp::Lte => lte::NAME,
            ast::BinOp::Gte => gte::NAME,
            ast::BinOp::And => and::NAME,
            ast::BinOp::Or => or::NAME,
        };

        self.scope(2, |vis, regs| {
            let [lhs, rhs] = regs else { unreachable!() };
            let lhs_code = vis.gen_expr_id(expr.lhs, lhs.clone().into(), doc);
            let rhs_code = vis.gen_expr_id(expr.rhs, rhs.clone().into(), doc);

            doc.stmts([
                lhs_code,
                rhs_code,
                doc.assign_place(
                    result_var,
                    doc.call_expr(
                        doc.text(op_builtin),
                        [lhs.clone().pretty(doc), rhs.clone().pretty(doc)],
                    ),
                ),
            ])
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
            ast::LitExpr::Tuple(tup) => self.scope(2, |vis, regs| {
                let [fst, snd] = regs else { unreachable!() };
                let fst_code = vis.gen_expr_id(tup.first, fst.clone(), doc);
                let snd_code = vis.gen_expr_id(tup.second, snd.clone(), doc);

                doc.stmts([
                    fst_code,
                    snd_code,
                    doc.assign_place(
                        result_var,
                        doc.builtin_call2(
                            mk_tuple,
                            fst.clone().pretty(doc),
                            snd.clone().pretty(doc),
                        ),
                    ),
                ])
            }),
        }
    }

    pub fn gen_call_expr(
        &mut self,
        expr: &ast::CallExpr,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
        self.scope(2, |vis, regs| {
            let [callee, args_var] = regs else { unreachable!() };

            let callee_code = vis.gen_expr_id(expr.callee, callee.clone(), doc);
            let args_code = doc.stmts(expr.args.iter().enumerate().map(|(i, &arg)| {
                vis.scope(1, |vis, regs| {
                    let code = vis.gen_expr_id(arg, regs[0].clone(), doc);
                    doc.stmts([
                        code,
                        // set_arg(tmp, 0, arg0);
                        doc.stmt(doc.builtin_call3(
                            set_arg,
                            args_var.clone().pretty(doc),
                            doc.as_string(i),
                            regs[0].clone().pretty(doc),
                        )),
                    ])
                })
            }));

            doc.stmts([
                callee_code,
                // tmp = mk_args(N);
                doc.assign_place(
                    args_var.clone(),
                    doc.builtin_call1(mk_args, doc.as_string(expr.args.len())),
                ),
                args_code,
                // result = call(callee, result /* the arg list */);
                doc.assign_place(
                    result_var,
                    doc.builtin_call2(call, callee.clone().pretty(doc), args_var.clone().pretty(doc)),
                ),
            ])
        })
    }

    pub fn gen_builtin_expr(
        &mut self,
        expr: &ast::BuiltinExpr,
        result_var: Place,
        doc: &'a DocAlloc<'a>,
    ) -> DocBuilder<'a> {
        self.scope(expr.args.len(), |vis, regs| {
            let args_code = doc.stmts(
                expr.args
                    .iter()
                    .zip(regs)
                    .map(|(&arg, var)| vis.gen_expr_id(arg, var.clone(), doc)),
            );

            let fn_name = doc.as_string(&vis.cx[expr.name]).double_quotes();
            let args_var = vis.tmp_var();

            doc.stmts([
                args_code,
                // result = mk_args(N);
                doc.assign_place(
                    args_var.clone(),
                    doc.builtin_call1(mk_args, doc.as_string(expr.args.len())),
                ),
                // set_arg(result, 0, arg0);
                // set_arg(result, 1, arg1);
                // ...and so on
                doc.stmts(regs.iter().enumerate().map(|(i, reg)| {
                    doc.stmt(doc.builtin_call3(
                        set_arg,
                        args_var.clone().pretty(doc),
                        doc.as_string(i),
                        reg.clone().pretty(doc),
                    ))
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
    mk_closure/2;

    mk_var_uninit/0;
    var_init/2;

    read_bool/1;

    closure_obj/1;

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
    rem/2;
    eq/2;
    neq/2;
    lt/2;
    lte/2;
    gt/2;
    gte/2;
    and/2;
    or/2;
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

    fn bin_expr(
        &'a self,
        lhs: DocBuilder<'a>,
        op: DocBuilder<'a>,
        rhs: DocBuilder<'a>,
    ) -> DocBuilder<'a> {
        self.intersperse([lhs, op, rhs], self.space())
    }

    fn assign_stmt(
        &'a self,
        var: impl Pretty<'a, Self>,
        value: impl Pretty<'a, Self>,
    ) -> DocBuilder<'a> {
        self.stmt(docs![self, var, self.reflow(" = "), value])
    }

    fn assign_place(
        &'a self,
        place: impl Into<Place>,
        value: impl Pretty<'a, Self>,
    ) -> DocBuilder<'a> {
        let place = place.into();
        match &place {
            Place::Register(_) => self.assign_stmt(place, value),
            Place::Var(_) => {
                self.stmt(self.builtin_call2(var_init, place.pretty(self), value.pretty(self)))
            }
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
        self.stmt(docs![self, "return ", value])
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
