use serde_json::value as json;

use crate::{ast, VisitContext};

pub fn parse<T, Context>(value: &json::Value, cx: &mut Context) -> Result<T, Error>
where
    T: FromJson<Context>,
{
    FromJson::from_json(Value(value), cx)
}

pub trait FromJson<Context>: Sized {
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error>;
}

#[derive(Debug, Clone)]
pub enum Error {
    ExpectedObject,
    ExpectedArray,
    ExpectedString,
    ExpectedInt,
    ExpectedFloat,
    ExpectedBool,
    UnexpectedTag { tag: String },
    MissingField { name: &'static str },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ExpectedObject => write!(f, "expected object"),
            Error::ExpectedArray => write!(f, "expected array"),
            Error::ExpectedString => write!(f, "expected string"),
            Error::ExpectedInt => write!(f, "expected int"),
            Error::ExpectedFloat => write!(f, "expected float"),
            Error::ExpectedBool => write!(f, "expected bool"),
            Error::UnexpectedTag { tag } => write!(f, "unexpected tag '{tag}'"),
            Error::MissingField { name } => write!(f, "missing field '{name}'"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy)]
pub struct Value<'a>(pub &'a json::Value);

impl<'a> Value<'a> {
    pub fn object(self) -> Result<Object<'a>, Error> {
        let json::Value::Object(obj) = self.0 else {
            return Err(Error::ExpectedObject);
        };
        Ok(Object(obj))
    }

    pub fn array(self) -> Result<Array<'a>, Error> {
        let json::Value::Array(arr) = self.0 else {
            return Err(Error::ExpectedArray);
        };
        Ok(Array(arr))
    }

    pub fn string(self) -> Result<&'a str, Error> {
        let json::Value::String(s) = self.0 else {
            return Err(Error::ExpectedString);
        };
        Ok(s.as_str())
    }

    pub fn i64(self) -> Result<i64, Error> {
        self.0.as_i64().ok_or_else(|| Error::ExpectedInt)
    }

    pub fn f64(self) -> Result<f64, Error> {
        self.0.as_f64().ok_or_else(|| Error::ExpectedFloat)
    }

    pub fn usize(self) -> Result<usize, Error> {
        // TODO: improve error
        self.i64()?.try_into().map_err(|_| Error::ExpectedInt)
    }

    pub fn bool(self) -> Result<bool, Error> {
        self.0.as_bool().ok_or_else(|| Error::ExpectedBool)
    }

    pub fn parse<T, Context>(self, cx: &mut Context) -> Result<T, Error>
    where
        T: FromJson<Context>,
    {
        T::from_json(self, cx)
    }
}

impl<'a> From<&'a json::Value> for Value<'a> {
    fn from(value: &'a json::Value) -> Self {
        Value(value)
    }
}

pub struct Object<'a>(pub &'a json::Map<String, json::Value>);

impl<'a> Object<'a> {
    pub fn field(&self, name: &'static str) -> Result<Value<'a>, Error> {
        self.0
            .get(name)
            .map(Value)
            .ok_or_else(|| Error::MissingField { name })
    }
}

pub struct Array<'a>(pub &'a [json::Value]);

impl<'a> Array<'a> {
    pub fn iter(&self) -> impl Iterator<Item = Value<'a>> {
        self.0.iter().map(Value)
    }
}

/* -- Helper traits -- */

pub trait ExprAlloc {
    fn alloc(&mut self, expr: ast::Expr) -> ast::ExprId;
}

pub trait StrInterner {
    fn intern_str(&mut self, s: &str) -> ast::InternedStr;

    fn get_str(&self, reference: ast::InternedStr) -> &str;

    fn intern_static(&mut self, s: &'static str) -> ast::InternedStr {
        self.intern_str(s)
    }
}

pub trait LocationAlloc {
    fn alloc_loc(&mut self, loc: ast::LocationData) -> ast::Location;

    fn get_loc_data(&self, reference: ast::Location) -> &ast::LocationData;
}

/* -- Basic context -- */

#[derive(Debug, Clone)]
pub struct BasicContext {
    pub exprs: Vec<ast::Expr>,
    pub strs: indexmap::IndexSet<String>,
    pub locs: indexmap::IndexSet<ast::LocationData>,
}

impl BasicContext {
    pub fn new() -> Self {
        let mut cx = BasicContext {
            exprs: Vec::new(),
            strs: indexmap::IndexSet::new(),
            locs: indexmap::IndexSet::new(),
        };

        let (id, _) = cx.strs.insert_full(String::from("builtin"));
        let builtin_file = ast::InternedStr(id as u32);

        cx.locs.insert(ast::LocationData {
            start: 0,
            end: 0,
            file: builtin_file,
        });

        cx
    }

    pub fn ident(&self, s: &str) -> Option<ast::Ident> {
        let ix = self.strs.get_index_of(s)?;
        Some(ast::Ident(ast::InternedStr(ix as u32)))
    }
}

impl std::ops::Index<ast::ExprId> for BasicContext {
    type Output = ast::Expr;

    fn index(&self, index: ast::ExprId) -> &Self::Output {
        &self.exprs[index.0 as usize]
    }
}

impl std::ops::Index<ast::Ident> for BasicContext {
    type Output = str;

    fn index(&self, index: ast::Ident) -> &Self::Output {
        &self[index.0]
    }
}

impl std::ops::Index<ast::InternedStr> for BasicContext {
    type Output = str;

    fn index(&self, index: ast::InternedStr) -> &Self::Output {
        self.strs[index.0 as usize].as_str()
    }
}

impl ExprAlloc for BasicContext {
    fn alloc(&mut self, expr: ast::Expr) -> ast::ExprId {
        let id = self.exprs.len();
        self.exprs.push(expr);
        ast::ExprId(id as u32)
    }
}

impl StrInterner for BasicContext {
    fn intern_str(&mut self, s: &str) -> ast::InternedStr {
        if let Some(id) = self.strs.get_index_of(s) {
            ast::InternedStr(id as u32)
        } else {
            let (id, success) = self.strs.insert_full(s.to_owned());
            debug_assert!(success);
            ast::InternedStr(id as u32)
        }
    }

    fn get_str(&self, reference: ast::InternedStr) -> &str {
        &self[reference]
    }
}

impl LocationAlloc for BasicContext {
    fn alloc_loc(&mut self, loc: ast::LocationData) -> ast::Location {
        let (id, _success) = self.locs.insert_full(loc);
        ast::Location(id as u32)
    }

    fn get_loc_data(&self, reference: ast::Location) -> &ast::LocationData {
        self.locs.get_index(reference.0 as usize).expect("invalid location reference")
    }
}

impl VisitContext for BasicContext {
    fn get_expr(&self, id: ast::ExprId) -> &ast::Expr {
        &self[id]
    }
}

/* -- AST impls -- */

impl<Context> FromJson<Context> for ast::File
where
    Context: ExprAlloc,
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(ast::File {
            name: obj.field("name")?.string()?.to_owned(),
            expr: obj.field("expression")?.parse(cx)?,
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::Expr
where
    Context: ExprAlloc,
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;
        let tag = obj.field("kind")?.string()?;

        match tag {
            "Function" => Ok(Self::Fn(value.parse(cx)?)),
            "If" => Ok(Self::If(value.parse(cx)?)),
            "Let" => Ok(Self::Let(value.parse(cx)?)),
            "Binary" => Ok(Self::Bin(value.parse(cx)?)),
            "Var" => Ok(Self::Var(value.parse(cx)?)),
            "Call" => Ok(Self::Call(value.parse(cx)?)),
            "Str" | "Int" | "Bool" | "Tuple" => Ok(Self::Lit(value.parse(cx)?)),
            "Print" | "First" | "Second" => Ok(Self::Builtin(value.parse(cx)?)),
            _ => Err(Error::UnexpectedTag { tag: tag.into() }),
        }
    }
}

impl<Context> FromJson<Context> for ast::BinExpr
where
    Context: ExprAlloc,
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(ast::BinExpr {
            op: obj.field("op")?.parse(cx)?,
            lhs: obj.field("lhs")?.parse(cx)?,
            rhs: obj.field("rhs")?.parse(cx)?,
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::BinOp {
    fn from_json<'a>(value: Value<'a>, _cx: &mut Context) -> Result<Self, Error> {
        let op = value.string()?;

        match op {
            "Add" => Ok(ast::BinOp::Add),
            "Sub" => Ok(ast::BinOp::Sub),
            "Mul" => Ok(ast::BinOp::Mul),
            "Div" => Ok(ast::BinOp::Div),
            "Rem" => Ok(ast::BinOp::Rem),
            "Eq" => Ok(ast::BinOp::Eq),
            "Neq" => Ok(ast::BinOp::Neq),
            "Lt" => Ok(ast::BinOp::Lt),
            "Gt" => Ok(ast::BinOp::Gt),
            "Lte" => Ok(ast::BinOp::Lte),
            "Gte" => Ok(ast::BinOp::Gte),
            "And" => Ok(ast::BinOp::And),
            "Or" => Ok(ast::BinOp::Or),
            _ => Err(Error::UnexpectedTag { tag: op.to_owned() }),
        }
    }
}

impl<Context> FromJson<Context> for ast::VarExpr
where
    Context: ExprAlloc,
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(ast::VarExpr {
            ident: obj.field("text")?.parse(cx)?,
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::FnExpr
where
    Context: ExprAlloc,
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(ast::FnExpr {
            params: obj.field("parameters")?.parse(cx)?,
            body: obj.field("value")?.parse(cx)?,
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::IfExpr
where
    Context: ExprAlloc,
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(ast::IfExpr {
            condition: obj.field("condition")?.parse(cx)?,
            then: obj.field("then")?.parse(cx)?,
            otherwise: obj.field("otherwise")?.parse(cx)?,
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::CallExpr
where
    Context: ExprAlloc,
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(ast::CallExpr {
            callee: obj.field("callee")?.parse(cx)?,
            args: obj.field("arguments")?.parse(cx)?,
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::LetExpr
where
    Context: ExprAlloc,
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(ast::LetExpr {
            name: obj.field("name")?.parse(cx)?,
            init: obj.field("value")?.parse(cx)?,
            next: obj.field("next")?.parse(cx)?,
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::BuiltinExpr
where
    Context: ExprAlloc,
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(Self {
            name: obj.field("kind")?.parse(cx)?,
            args: vec![obj.field("value")?.parse(cx)?],
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::LitExpr
where
    Context: ExprAlloc,
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;
        let tag = obj.field("kind")?.string()?;

        match tag {
            "Str" => Ok(Self::Str(value.parse(cx)?)),
            "Int" => Ok(Self::Int(value.parse(cx)?)),
            "Bool" => Ok(Self::Bool(value.parse(cx)?)),
            "Tuple" => Ok(Self::Tuple(value.parse(cx)?)),
            _ => Err(Error::UnexpectedTag {
                tag: tag.to_owned(),
            }),
        }
    }
}

impl<Context> FromJson<Context> for ast::StrLit
where
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(Self {
            value: obj.field("value")?.string()?.to_owned(),
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::IntLit
where
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(Self {
            value: obj.field("value")?.i64()?.to_owned(),
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::BoolLit
where
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(Self {
            value: obj.field("value")?.bool()?.to_owned(),
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::TupleLit
where
    Context: ExprAlloc,
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(Self {
            first: obj.field("first")?.parse(cx)?,
            second: obj.field("second")?.parse(cx)?,
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::Param
where
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(Self {
            ident: obj.field("text")?.parse(cx)?,
            loc: obj.field("location")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::Location
where
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let loc = value.parse(cx)?;
        Ok(cx.alloc_loc(loc))
    }
}

impl<Context> FromJson<Context> for ast::LocationData
where
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let obj = value.object()?;

        Ok(Self {
            start: obj.field("start")?.usize()?,
            end: obj.field("end")?.usize()?,
            file: obj.field("filename")?.parse(cx)?,
        })
    }
}

impl<Context> FromJson<Context> for ast::ExprId
where
    Context: ExprAlloc,
    Context: StrInterner,
    Context: LocationAlloc,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        let expr = value.parse(cx)?;
        Ok(cx.alloc(expr))
    }
}

impl<Context> FromJson<Context> for ast::Ident
where
    Context: StrInterner,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        Ok(ast::Ident(value.parse(cx)?))
    }
}

impl<Context> FromJson<Context> for ast::InternedStr
where
    Context: StrInterner,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        Ok(cx.intern_str(value.string()?))
    }
}

impl<T, Context> FromJson<Context> for Vec<T>
where
    T: FromJson<Context>,
{
    fn from_json<'a>(value: Value<'a>, cx: &mut Context) -> Result<Self, Error> {
        value.array()?.iter().map(|el| el.parse(cx)).collect()
    }
}
