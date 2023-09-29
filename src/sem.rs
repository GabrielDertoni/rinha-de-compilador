
use std::{collections::HashMap, rc::Rc};

use ast::{from_json::BasicContext, AstNode};

#[derive(Debug, Clone)]
pub enum Type {
    Str,
    Int,
    Bool,
    Tuple(TypeId, TypeId),
    Closure {
        params: Vec<TypeId>,
        ret: TypeId,
    },
    Unknown,
}

#[derive(Debug, Clone, Copy)]
pub enum DefKind {
    Let,
    LetFn,
    Param,
}

#[derive(Debug, Clone, Copy)]
pub enum RefKind {
    Local,
    Captured { scope: ast::ExprId },
    Param { func: ast::ExprId },
}

#[derive(Debug, Clone)]
pub struct Def {
    pub kind: DefKind,
    pub let_expr: ast::ExprId,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TypeId(pub u32);

#[derive(Debug, Clone)]
pub struct SemanticContext {
    table: InfoTable,
}

impl SemanticContext {
    pub fn new(cx: Rc<BasicContext>) -> Self {
        SemanticContext { table: InfoTable::new(cx) }
    }

    pub fn type_of(&self, expr: ast::ExprId) -> Type {
        todo!()
    }

    pub fn var_info(&self, expr: ast::ExprId) -> &VarInfo {
        assert!(self.table.cx[expr].as_var().is_some(), "id should refer to a variable expression");
        self.table.infos[&expr].var.as_ref().unwrap()
    }

    pub fn def_info(&self, expr: ast::ExprId) -> &Def {
        assert!(self.table.cx[expr].as_let().is_some(), "id should refer to a variable expression");
        self.table.infos[&expr].def.as_ref().unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct InfoTable {
    infos: HashMap<ast::ExprId, Info>,
    types: indexmap::IndexMap<TypeId, Type>,
    cx: Rc<BasicContext>,
}

impl InfoTable {
    pub fn new(cx: Rc<BasicContext>) -> Self {
        InfoTable {
            infos: HashMap::new(),
            types: indexmap::IndexMap::new(),
            cx,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Info {
    pub free_vars: Option<HashMap<ast::Ident, ast::ExprId>>,

    // If it is a `VarExpr`. If not, will always be `None`.
    pub var: Option<VarInfo>,
    pub def: Option<Def>,
}

#[derive(Debug, Clone)]
pub struct VarInfo {
    pub ref_kind: RefKind,
    pub def: Def,
}

impl InfoTable {
    /*
    pub fn query_free_vars(&mut self, id: ast::ExprId) -> &HashMap<ast::Ident, ast::ExprId> {
        let info = self.infos.entry(id)
            .or_insert_with(Default::default);

        if info.free_vars.is_none() {
            let mut vis = FreeVars::new();
            vis.visit_expr_id(id);
            info.free_vars = Some(vis.finish());
        }

        info.free_vars.as_ref().unwrap()
    }
    */
}

#[derive(Debug, Clone)]
pub struct FreeVars {
    defs: indexmap::IndexSet<ast::Ident>,
    vars: HashMap<ast::Ident, ast::ExprId>,
    curr: Option<ast::ExprId>,
}

impl FreeVars {
    pub fn new() -> Self {
        FreeVars {
            defs: indexmap::IndexSet::new(),
            vars: HashMap::new(),
            curr: None,
        }
    }

    pub fn finish(self) -> HashMap<ast::Ident, ast::ExprId> {
        self.vars
    }
}

impl<Context: ast::VisitContext> ast::Visitor<Context> for FreeVars {
    fn visit_expr_id(&mut self, &id: &ast::ExprId, cx: &Context) {
        let save = self.curr;
        self.curr = Some(id);
        id.visit_children(self, cx);
        self.curr = save;
    }

    fn visit_var_expr(&mut self, expr: &ast::VarExpr, _cx: &Context) {
        if !self.defs.contains(&expr.ident) {
            // Found a free variable
            self.vars.insert(expr.ident, self.curr.unwrap());
        }
    }

    fn visit_let_expr(&mut self, expr: &ast::LetExpr, cx: &Context) {
        expr.init.accept(self, cx);
        self.defs.insert(expr.name.ident);
        expr.next.accept(self, cx);
        self.defs.pop();
    }

    fn visit_fn_expr(&mut self, expr: &ast::FnExpr, cx: &Context) {
        let save = self.defs.len();

        // Push the stack frame
        for param in &expr.params {
            self.defs.insert(param.ident);
        }

        expr.visit_children(self, cx);

        // Pop the stack frame
        while self.defs.len() > save {
            self.defs.pop();
        }
    }
}
