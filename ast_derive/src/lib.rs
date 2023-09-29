use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::{parse_macro_input, DeriveInput, spanned::Spanned};

#[proc_macro_derive(AstNode, attributes(visit, visit_skip_all, visit_skip))]
pub fn derive_ast_node(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let impl_tokens = match &input.data {
        syn::Data::Struct(data) => ast_node_struct(&name, &input.attrs, data),
        syn::Data::Enum(data) => ast_node_enum(&name, &input.attrs, data),
        syn::Data::Union(_) => unimplemented!(),
    };

    impl_tokens.into()
}

fn ast_node_enum(name: &syn::Ident, attrs: &[syn::Attribute], data: &syn::DataEnum) -> TokenStream {
    let visitor = format_ident!("visitor");
    let cx = format_ident!("cx");

    let Some(attr) = attrs.iter().find(|attr| attr.path().is_ident("visit")) else {
        panic!("expected #[visit] attribute");
    };
    let visit_fn: syn::Path = attr.parse_args().unwrap();
    let visit_children = !attrs.iter().any(|attr| attr.path().is_ident("visit_skip_all"));
    let skiped_fields = get_skipped_fields(attrs);

    let children = data.variants.iter().map(|variant| {
        let variant_name = &variant.ident;
        let children = call_accpet_on_fields(&visitor, &cx, &skiped_fields, &variant.fields);
        let destruct = destruct_fields(&variant.fields);
        let visit_variant = !variant.attrs.iter().any(|attr| attr.path().is_ident("visit_skip"))
            && !skiped_fields.contains(&variant.ident.to_string());
        if visit_children && visit_variant {
            quote_spanned!(variant.span() => #name::#variant_name #destruct => { #(#children;)* })
        } else {
            quote!(#name::#variant_name(..) => ())
        }
    });

    quote! {
        #[allow(unused_variables)]
        impl AstNode for #name {
            fn accept<V, Cx>(&self, #visitor: &mut V, #cx: &Cx)
            where
                V: ast::Visitor<Cx> + ?Sized,
                Cx: ast::VisitContext + ?Sized,
            {
                visitor.#visit_fn(self, #cx);
            }

            fn visit_children<V, Cx>(&self, #visitor: &mut V, #cx: &Cx)
            where
                V: ast::Visitor<Cx> + ?Sized,
                Cx: ast::VisitContext + ?Sized,
            {
                match self {
                    #(#children,)*
                }
            }
        }
    }
}

fn ast_node_struct(
    name: &syn::Ident,
    attrs: &[syn::Attribute],
    data: &syn::DataStruct,
) -> TokenStream {
    let visitor = format_ident!("visitor");
    let cx = format_ident!("cx");

    let Some(attr) = attrs.iter().find(|attr| attr.path().is_ident("visit")) else {
        panic!("expected #[visit] attribute");
    };
    let visit_fn: syn::Path = attr.parse_args().unwrap();
    let visit_children = !attrs.iter().any(|attr| attr.path().is_ident("visit_skip_all"));
    let skiped_fields = get_skipped_fields(attrs);

    let destruct = destruct_fields(&data.fields);
    let children = if visit_children {
        let children = call_accpet_on_fields(&visitor, &cx, &skiped_fields, &data.fields);
        quote!(#(#children;)*)
    } else {
        quote!()
    };

    quote! {
        #[allow(unused_variables)]
        impl ast::AstNode for #name {
            fn accept<V, Cx>(&self, #visitor: &mut V, #cx: &Cx)
            where
                V: ast::Visitor<Cx> + ?Sized,
                Cx: ast::VisitContext + ?Sized,
            {
                visitor.#visit_fn(self, #cx);
            }

            fn visit_children<V, Cx>(&self, #visitor: &mut V, #cx: &Cx)
            where
                V: ast::Visitor<Cx> + ?Sized,
                Cx: ast::VisitContext + ?Sized,
            {
                let Self #destruct = self;
                #children
            }
        }
    }
}

// Will assume that the fields are already destructed
fn call_accpet_on_fields<'a>(
    visitor: &'a syn::Ident,
    cx: &'a syn::Ident,
    skiped_fields: &'a [String],
    fields: &'a syn::Fields,
) -> impl Iterator<Item = TokenStream> + 'a {
    fields.iter().enumerate().map(move |(i, field)| {
        let is_skiped = if let Some(ident) = &field.ident {
            skiped_fields.contains(&ident.to_string())
        } else {
            // NOTE(gd): cannot annotate to skip a field in a tuple struct
            false
        };

        let visit_field = !field.attrs.iter().any(|attr| attr.path().is_ident("visit_skip")) && !is_skiped;
        let field_ident = field
            .ident
            .clone()
            .unwrap_or_else(|| format_ident!("field{i}"));
        if visit_field {
            quote_spanned!(field.span() => #field_ident.accept(#visitor, #cx))
        } else {
            quote!()
        }
    })
}

fn destruct_fields(fields: &syn::Fields) -> TokenStream {
    match fields {
        syn::Fields::Named(named) => {
            let names = named
                .named
                .iter()
                .map(|field| field.ident.as_ref().unwrap());
            quote!({ #(#names),* })
        }
        syn::Fields::Unnamed(unnamed) => {
            let names = (0..unnamed.unnamed.len()).map(|i| format_ident!("field{i}"));
            quote!((#(#names),*))
        }
        syn::Fields::Unit => quote!(),
    }
}

fn get_skipped_fields(attrs: &[syn::Attribute]) -> Vec<String> {
    use syn::{Token, punctuated::Punctuated};

    type PunctuatedIdent = Punctuated::<syn::Ident, Token![,]>;

    attrs.iter()
        .filter(|attr| attr.path().is_ident("visit_skip"))
        .flat_map(|attr| attr.parse_args_with(PunctuatedIdent::parse_terminated).unwrap())
        .map(|id| id.to_string())
        .collect()
}
