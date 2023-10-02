#ifndef _RT_H
#define _RT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <dlfcn.h>
#include <assert.h>

typedef struct object_data* object_t;
typedef struct closure_data* closure_t;

typedef struct arg_list_data* args_t;

typedef object_t (*closure_base_t)(object_t*, object_t);
typedef object_t (*extern_fn_t)(args_t);

typedef enum {
    TYPE_STR,
    TYPE_STATIC_STR,
    TYPE_INT,
    TYPE_BOOL,
    TYPE_TUPLE,
    TYPE_CLOSURE,
    TYPE_NULL,
    TYPE_ARGS,

    TYPE_UNINIT,
} type_t;

// Null terminated for better compatibility
typedef struct {
    char* ptr;
    size_t len;
} str_t;

typedef struct {
    object_t first;
    object_t second;
} tuple_t;

typedef struct arg_list_data {
    int len;
    object_t args[];
} arg_list_t;

typedef struct closure_data {
    void* entry;
    object_t capture[];
} closure_data_t;

typedef union {
    str_t as_str;
    int64_t as_int;
    bool as_bool;
    closure_data_t as_closure;
    tuple_t as_tuple;
    arg_list_t as_args;
} data_t;

typedef struct object_data {
    type_t type;
    data_t data;
} object_data_t;

extern object_data_t null_object_data;
extern object_data_t true_object_data;
extern object_data_t false_object_data;

#define NULL_OBJECT (object_t)&null_object_data
#define TRUE_OBJECT (object_t)&true_object_data
#define FALSE_OBJECT (object_t)&false_object_data

/* -- RUNTIME -- */

#define RT(name) _rt_##name

void RT(init)();
void* RT(get_fn)(const char* name);

/* -- BUILTINS -- */

#define BUILTIN(name) _rt__builtin__##name

// mk_object/2;
object_t BUILTIN(mk_object)(type_t type, data_t data);

// mk_static_str/1;
object_t BUILTIN(mk_static_str)(const char* str);

// mk_int/1;
object_t BUILTIN(mk_int)(int value);

// mk_bool/1;
object_t BUILTIN(mk_bool)(bool value);

// mk_tuple/2;
object_t BUILTIN(mk_tuple)(object_t first, object_t second);

// call/2;
object_t BUILTIN(call)(object_t callee, object_t arg_list);

// extern_call/2;
object_t BUILTIN(extern_call)(const char* name, object_t arg_list);

// mk_arg_list/1;
object_t BUILTIN(mk_args)(int n);

// set_arg_list/3;
object_t BUILTIN(set_arg)(object_t arg_list, int i, object_t value);

// get_arg_list/2;
object_t BUILTIN(get_arg)(object_t arg_list, int i);

// mk_var_uninit/0;
object_t BUILTIN(mk_var_uninit)();

// mk_closure/2;
object_t* BUILTIN(mk_closure)(int n_fields, void* entry);

// closure_obj/1;
object_t BUILTIN(closure_obj)(object_t* capture);

// var_init/2;
object_t BUILTIN(var_init)(object_t var, object_t value);

// read_boo/1;
bool BUILTIN(read_bool)(object_t value);

// Arithmetic operations

object_t BUILTIN(lt)(object_t lhs, object_t rhs);
object_t BUILTIN(add)(object_t lhs, object_t rhs);
object_t BUILTIN(sub)(object_t lhs, object_t rhs);

#endif
