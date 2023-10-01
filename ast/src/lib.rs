extern crate self as ast;

mod impls;
mod tree;
pub mod from_json;

pub use ast_derive::AstNode;
pub use tree::*;

use std::ops::Index;

#[macro_export]
macro_rules! from_json {
    ($cx:ident, $($tt:tt)*) => { {
        let json = serde_json::json!($($tt)*);
        ast::from_json::FromJson::from_json((&json).into(), &mut $cx)
    } };
}

pub trait VisitContext {
    fn get_expr(&self, id: ExprId) -> &Expr;
}

pub trait AstNode {
    fn accept<V, Cx>(&self, visitor: &mut V, cx: &Cx)
    where
        V: Visitor<Cx> + ?Sized,
        Cx: VisitContext + ?Sized;

    fn visit_children<V, Cx>(&self, visitor: &mut V, cx: &Cx)
    where
        V: Visitor<Cx> + ?Sized,
        Cx: VisitContext + ?Sized;
}

impl<T: AstNode> AstNode for [T] {
    fn accept<V, Cx>(&self, visitor: &mut V, cx: &Cx)
    where
        V: Visitor<Cx> + ?Sized,
        Cx: VisitContext + ?Sized,
    {
        // FIXME: This seems wrong, because that is what `visit_children` is suposed to do
        for child in self {
            child.accept(visitor, cx)
        }
    }

    fn visit_children<V, Cx>(&self, visitor: &mut V, cx: &Cx)
    where
        V: Visitor<Cx> + ?Sized,
        Cx: VisitContext + ?Sized,
    {
        for child in self {
            child.accept(visitor, cx)
        }
    }
}

pub trait Visitor<Context: VisitContext + ?Sized> {
    fn visit_expr_id(&mut self, expr: &ExprId, cx: &Context) {
        expr.visit_children(self, cx);
    }

    fn visit_expr(&mut self, expr: &Expr, cx: &Context) {
        expr.visit_children(self, cx);
    }

    fn visit_var_expr(&mut self, expr: &VarExpr, cx: &Context) {
        expr.visit_children(self, cx);
    }

    fn visit_fn_expr(&mut self, expr: &FnExpr, cx: &Context) {
        expr.visit_children(self, cx);
    }

    fn visit_if_expr(&mut self, expr: &IfExpr, cx: &Context) {
        expr.visit_children(self, cx);
    }

    fn visit_let_expr(&mut self, expr: &LetExpr, cx: &Context) {
        expr.visit_children(self, cx);
    }

    fn visit_bin_expr(&mut self, expr: &BinExpr, cx: &Context) {
        expr.visit_children(self, cx);
    }

    fn visit_call_expr(&mut self, expr: &CallExpr, cx: &Context) {
        expr.visit_children(self, cx);
    }

    fn visit_lit_expr(&mut self, expr: &LitExpr, cx: &Context) {
        expr.visit_children(self, cx)
    }

    fn visit_builtin_expr(&mut self, expr: &BuiltinExpr, cx: &Context) {
        expr.visit_children(self, cx);
    }
}
