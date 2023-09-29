#include "rt.h"

// static, read only data

object_data_t null_object_data = {
    .type = TYPE_NULL,
    .data = {0},
};

object_data_t true_object_data = {
    .type = TYPE_BOOL,
    .data = { .as_bool = true },
};

object_data_t false_object_data = {
    .type = TYPE_BOOL,
    .data = { .as_bool = false },
};

/* -- RUNTIME -- */

static void* RT(extern_fns_handle) = NULL;

void RT(init)() {
    RT(extern_fns_handle) = dlopen("functions.so", RTLD_NOW);
    if (!RT(extern_fns_handle)) {
        fprintf(stderr, "failed to load functions\n");
        exit(1);
    }
}

void* RT(get_fn)(const char* name) {
    assert(RT(extern_fns_handle));
    return dlsym(RT(extern_fns_handle), name);
}

/* -- BUILTINS -- */

object_t BUILTIN(mk_object)(type_t type, data_t data) {
    object_t obj = (object_t)malloc(sizeof(object_data_t));
    obj->type = type;
    obj->data = data;
    return obj;
}

// mk_static_str/1;
object_t BUILTIN(mk_static_str)(const char* str) {
    data_t data = {
        .as_str = {
            .ptr = (char*)str,
            .len = strlen(str)
        }
    };
    return BUILTIN(mk_object)(TYPE_STATIC_STR, data);
}

// mk_int/1;
object_t BUILTIN(mk_int)(int value) {
    return BUILTIN(mk_object)(TYPE_INT, (data_t){.as_int = value});
}

// mk_bool/1;
object_t BUILTIN(mk_bool)(bool value) {
    return BUILTIN(mk_object)(TYPE_BOOL, (data_t){.as_bool = value});
}

// mk_tuple/2;
object_t BUILTIN(mk_tuple)(object_t first, object_t second) {
    data_t data = {
        .as_tuple = {
            .first = first,
            .second = second,
        }
    };
    return BUILTIN(mk_object)(TYPE_TUPLE, data);
}

// call/2;
object_t BUILTIN(call)(object_t callee, object_t arg_list) {
    if (callee->type != TYPE_CLOSURE) {
        fprintf(stderr, "rt error: expected callee to be closure, but was something else");
        exit(1);
    }
    closure_base_t fn_ptr = (closure_base_t)callee->data.as_closure.base;
    return fn_ptr(callee->data.as_closure.capture, arg_list);
}

// extern_call/2;
object_t BUILTIN(extern_call)(const char* name, object_t arg_list) {
    void* ptr = RT(get_fn)(name);
    if (!ptr) {
        fprintf(stderr, "function '%s' was not found", name);
        return NULL_OBJECT;
    }
    // TODO: exception
    assert(arg_list->type == TYPE_ARGS);
    extern_fn_t fn_ptr = (extern_fn_t)(ptr);
    return fn_ptr(&arg_list->data.as_args);
}

// mk_arg_list/1;
object_t BUILTIN(mk_arg_list)(int n) {
    object_t obj = (object_t)malloc(sizeof(object_data_t) + sizeof(object_t) * n);
    obj->type = TYPE_ARGS;
    obj->data.as_args.len = n;
    return obj;
}

// set_arg_list/3;
object_t BUILTIN(set_arg_list)(object_t arg_list, int i, object_t value) {
    // TODO: exception
    assert(arg_list->type == TYPE_ARGS);
    assert(i >= 0 && i < arg_list->data.as_args.len);

    arg_list->data.as_args.args[i] = value;
    return arg_list;
}

// get_arg_list/2;
object_t BUILTIN(get_arg_list)(object_t arg_list, int i) {
    // TODO: exception
    assert(arg_list->type == TYPE_ARGS);
    assert(i >= 0 && i < arg_list->data.as_args.len);

    return arg_list->data.as_args.args[i];
}

object_t BUILTIN(lt)(object_t lhs, object_t rhs) {
    // TODO: Throw exception
    assert(lhs->type == TYPE_INT);
    assert(rhs->type == TYPE_INT);

    if (lhs->data.as_int < rhs->data.as_int)
        return TRUE_OBJECT;
    else
        return FALSE_OBJECT;
}

object_t BUILTIN(add)(object_t lhs, object_t rhs) {
    // TODO: Throw exception
    assert(lhs->type == TYPE_INT);
    assert(rhs->type == TYPE_INT);

    data_t data = { .as_int = lhs->data.as_int + rhs->data.as_int };
    return BUILTIN(mk_object)(TYPE_INT, data);
}

object_t BUILTIN(sub)(object_t lhs, object_t rhs) {
    // TODO: Throw exception
    assert(lhs->type == TYPE_INT);
    assert(rhs->type == TYPE_INT);

    data_t data = { .as_int = lhs->data.as_int - rhs->data.as_int };
    return BUILTIN(mk_object)(TYPE_INT, data);
}
