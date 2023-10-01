use ast::*;

impl Index<ExprId> for std::rc::Rc<from_json::BasicContext> {
    type Output = ast::Expr;

    fn index(&self, index: ExprId) -> &Self::Output {
        self.as_ref().index(index)
    }
}

impl Index<Ident> for std::rc::Rc<from_json::BasicContext> {
    type Output = str;

    fn index(&self, index: Ident) -> &Self::Output {
        self.as_ref().index(index)
    }
}

impl VisitContext for std::rc::Rc<from_json::BasicContext> {
    fn get_expr(&self, id: ExprId) -> &Expr {
        self.as_ref().get_expr(id)
    }
}

impl Expr {
    pub fn as_var(&self) -> Option<&VarExpr> {
        if let Expr::Var(var) = self {
            Some(var)
        } else {
            None
        }
    }

    pub fn as_let(&self) -> Option<&LetExpr> {
        if let Expr::Let(expr) = self {
            Some(expr)
        } else {
            None
        }
    }
}

impl Location {
    pub fn builtin() -> Location {
        Location(0)
    }
}

impl AstNode for ExprId {
    fn accept<V, Cx>(&self, visitor: &mut V, cx: &Cx)
    where
        V: Visitor<Cx> + ?Sized,
        Cx: VisitContext + ?Sized,
    {
        visitor.visit_expr_id(self, cx)
    }

    fn visit_children<V, Cx>(&self, visitor: &mut V, cx: &Cx)
    where
        V: Visitor<Cx> + ?Sized,
        Cx: VisitContext + ?Sized,
    {
        cx.get_expr(*self).accept(visitor, cx)
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

impl std::fmt::Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ident")
    }
}

impl std::fmt::Display for ExprId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expr")
    }
}
