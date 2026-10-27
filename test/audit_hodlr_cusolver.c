/* Test-only LD_PRELOAD audit of actual native cuSOLVER resource creation. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

static _Atomic unsigned long handles, parameters, failures;

static void* symbol(const char* name) {
    void* result = dlsym(RTLD_NEXT, name);
    if (!result) { fprintf(stderr, "cuSOLVER audit: %s unavailable\n", name); abort(); }
    return result;
}
int cusolverDnCreate(void** handle) {
    int (*create)(void**) = (int (*)(void**))symbol("cusolverDnCreate");
    const int status = create(handle);
    atomic_fetch_add(status ? &failures : &handles, 1);
    return status;
}
int cusolverDnCreateParams(void** params) {
    int (*create)(void**) = (int (*)(void**))symbol("cusolverDnCreateParams");
    const int status = create(params);
    atomic_fetch_add(status ? &failures : &parameters, 1);
    return status;
}
__attribute__((destructor)) static void report(void) {
    const char* prefix = getenv("HODLR_CUSOLVER_AUDIT");
    const char* rank = getenv("SLURM_PROCID");
    if (!prefix || !rank) return;
    if (!atomic_load(&handles) && !atomic_load(&parameters) && !atomic_load(&failures)) return;
    char path[PATH_MAX];
    const int length = snprintf(path, sizeof(path), "%s.rank%s.json", prefix, rank);
    if (length < 0 || (size_t)length >= sizeof(path)) abort();
    FILE* file = fopen(path, "w");
    if (!file) abort();
    fprintf(file, "{\"handles_created\":%lu,\"params_created\":%lu,\"creation_failures\":%lu}\n",
            atomic_load(&handles), atomic_load(&parameters), atomic_load(&failures));
    fclose(file);
}
