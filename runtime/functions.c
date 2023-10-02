#include <assert.h>
#include <stdio.h>

#include "rt.h"

static object_t _print(object_t obj) {
    switch (obj->type) {
        case TYPE_STATIC_STR:
        case TYPE_STR:
            printf("%s\n", obj->data.as_str.ptr);
            break;
        case TYPE_INT:
            printf("%lld\n", obj->data.as_int);
            break;
        case TYPE_BOOL:
            if (obj->data.as_bool) printf("true\n");
            else printf("false\n");
            break;
        case TYPE_TUPLE:
            _print(obj->data.as_tuple.first);
            _print(obj->data.as_tuple.second);
            break;
        case TYPE_CLOSURE:
            printf("<#closure>\n");
            break;
        case TYPE_NULL:
            printf("null\n");
        case TYPE_ARGS:
            printf("args(");

            if (obj->data.as_args.len > 0)
                _print(obj->data.as_args.args[0]);

            for (int i = 1; i < obj->data.as_args.len; i++) {
                printf(", ");
                _print(obj->data.as_args.args[1]);
            }

            printf(")\n");
            break;
        default:
            // should be unreachable
            exit(1);
    }
    return obj;
}

object_t Print(args_t args) {
    // TODO: exception
    assert(args->len == 1);
    return _print(args->args[0]);
}
