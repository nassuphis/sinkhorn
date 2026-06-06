#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int waiting;
    int generation;
} simple_barrier_t;

static int barrier_init(simple_barrier_t *barrier, int count) {
    barrier->count = count;
    barrier->waiting = 0;
    barrier->generation = 0;
    if (pthread_mutex_init(&barrier->mutex, NULL) != 0) return -1;
    if (pthread_cond_init(&barrier->cond, NULL) != 0) return -1;
    return 0;
}

static void barrier_destroy(simple_barrier_t *barrier) {
    pthread_cond_destroy(&barrier->cond);
    pthread_mutex_destroy(&barrier->mutex);
}

static void barrier_wait(simple_barrier_t *barrier) {
    pthread_mutex_lock(&barrier->mutex);
    int generation = barrier->generation;
    barrier->waiting += 1;
    if (barrier->waiting == barrier->count) {
        barrier->generation += 1;
        barrier->waiting = 0;
        pthread_cond_broadcast(&barrier->cond);
    } else {
        while (generation == barrier->generation) {
            pthread_cond_wait(&barrier->cond, &barrier->mutex);
        }
    }
    pthread_mutex_unlock(&barrier->mutex);
}

typedef struct {
    const double *cost;
    double *plan;
    double *kernel;
    double *u;
    double *v;
    double *thread_cost;
    double *thread_row_error;
    double *thread_col_error;
    double epsilon;
    int n;
    int iterations;
    int threads;
    simple_barrier_t barrier;
} sinkhorn_shared_t;

typedef struct {
    sinkhorn_shared_t *shared;
    int id;
} worker_arg_t;

static int chunk_start(int total, int chunks, int id) {
    return (int)(((long long)total * id) / chunks);
}

static int chunk_end(int total, int chunks, int id) {
    return (int)(((long long)total * (id + 1)) / chunks);
}

static void *sinkhorn_worker(void *raw_arg) {
    worker_arg_t *arg = (worker_arg_t *)raw_arg;
    sinkhorn_shared_t *s = arg->shared;
    int id = arg->id;
    int n = s->n;
    int row_start = chunk_start(n, s->threads, id);
    int row_end = chunk_end(n, s->threads, id);
    int col_start = row_start;
    int col_end = row_end;
    int flat_start = chunk_start(n * n, s->threads, id);
    int flat_end = chunk_end(n * n, s->threads, id);
    double a = 1.0 / (double)n;

    for (int idx = flat_start; idx < flat_end; ++idx) {
        s->kernel[idx] = exp(-s->cost[idx] / s->epsilon);
    }

    for (int i = row_start; i < row_end; ++i) {
        s->u[i] = 1.0;
    }
    for (int j = col_start; j < col_end; ++j) {
        s->v[j] = 1.0;
    }

    barrier_wait(&s->barrier);

    for (int iter = 0; iter < s->iterations; ++iter) {
        for (int i = row_start; i < row_end; ++i) {
            const double *krow = s->kernel + ((size_t)i * (size_t)n);
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += krow[j] * s->v[j];
            }
            s->u[i] = a / (sum > DBL_MIN ? sum : DBL_MIN);
        }

        barrier_wait(&s->barrier);

        for (int j = col_start; j < col_end; ++j) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                sum += s->kernel[(size_t)i * (size_t)n + (size_t)j] * s->u[i];
            }
            s->v[j] = a / (sum > DBL_MIN ? sum : DBL_MIN);
        }

        barrier_wait(&s->barrier);
    }

    double local_cost = 0.0;
    double local_row_error = 0.0;
    for (int i = row_start; i < row_end; ++i) {
        double row_sum = 0.0;
        size_t base = (size_t)i * (size_t)n;
        for (int j = 0; j < n; ++j) {
            double p = s->u[i] * s->kernel[base + (size_t)j] * s->v[j];
            if (s->plan != NULL) {
                s->plan[base + (size_t)j] = p;
            }
            row_sum += p;
            local_cost += p * s->cost[base + (size_t)j];
        }
        double err = fabs(row_sum - a);
        if (err > local_row_error) local_row_error = err;
    }
    s->thread_cost[id] = local_cost;
    s->thread_row_error[id] = local_row_error;

    double local_col_error = 0.0;
    for (int j = col_start; j < col_end; ++j) {
        double col_sum = 0.0;
        for (int i = 0; i < n; ++i) {
            col_sum += s->u[i] * s->kernel[(size_t)i * (size_t)n + (size_t)j] * s->v[j];
        }
        double err = fabs(col_sum - a);
        if (err > local_col_error) local_col_error = err;
    }
    s->thread_col_error[id] = local_col_error;

    return NULL;
}

int sinkhorn_mt_run(
    const double *cost,
    int n,
    double epsilon,
    int iterations,
    int threads,
    double *plan,
    double *out_cost,
    double *out_row_error,
    double *out_col_error
) {
    if (cost == NULL || n <= 0 || epsilon <= 0.0 || iterations <= 0 || threads <= 0) {
        return -1;
    }
    if (threads > n) threads = n;

    sinkhorn_shared_t shared;
    shared.cost = cost;
    shared.plan = plan;
    shared.epsilon = epsilon;
    shared.n = n;
    shared.iterations = iterations;
    shared.threads = threads;

    size_t matrix_len = (size_t)n * (size_t)n;
    shared.kernel = (double *)malloc(matrix_len * sizeof(double));
    shared.u = (double *)malloc((size_t)n * sizeof(double));
    shared.v = (double *)malloc((size_t)n * sizeof(double));
    shared.thread_cost = (double *)calloc((size_t)threads, sizeof(double));
    shared.thread_row_error = (double *)calloc((size_t)threads, sizeof(double));
    shared.thread_col_error = (double *)calloc((size_t)threads, sizeof(double));

    if (
        shared.kernel == NULL ||
        shared.u == NULL ||
        shared.v == NULL ||
        shared.thread_cost == NULL ||
        shared.thread_row_error == NULL ||
        shared.thread_col_error == NULL
    ) {
        free(shared.kernel);
        free(shared.u);
        free(shared.v);
        free(shared.thread_cost);
        free(shared.thread_row_error);
        free(shared.thread_col_error);
        return -2;
    }

    if (barrier_init(&shared.barrier, threads) != 0) {
        free(shared.kernel);
        free(shared.u);
        free(shared.v);
        free(shared.thread_cost);
        free(shared.thread_row_error);
        free(shared.thread_col_error);
        return -3;
    }

    pthread_t *pthread_ids = (pthread_t *)malloc((size_t)threads * sizeof(pthread_t));
    worker_arg_t *args = (worker_arg_t *)malloc((size_t)threads * sizeof(worker_arg_t));
    if (pthread_ids == NULL || args == NULL) {
        barrier_destroy(&shared.barrier);
        free(pthread_ids);
        free(args);
        free(shared.kernel);
        free(shared.u);
        free(shared.v);
        free(shared.thread_cost);
        free(shared.thread_row_error);
        free(shared.thread_col_error);
        return -4;
    }

    int status = 0;
    int created = 0;
    for (int t = 0; t < threads; ++t) {
        args[t].shared = &shared;
        args[t].id = t;
        if (pthread_create(&pthread_ids[t], NULL, sinkhorn_worker, &args[t]) != 0) {
            status = -5;
            break;
        }
        created += 1;
    }

    for (int t = 0; t < created; ++t) {
        pthread_join(pthread_ids[t], NULL);
    }

    if (status == 0) {
        double total_cost = 0.0;
        double row_error = 0.0;
        double col_error = 0.0;
        for (int t = 0; t < threads; ++t) {
            total_cost += shared.thread_cost[t];
            if (shared.thread_row_error[t] > row_error) row_error = shared.thread_row_error[t];
            if (shared.thread_col_error[t] > col_error) col_error = shared.thread_col_error[t];
        }
        if (out_cost != NULL) *out_cost = total_cost;
        if (out_row_error != NULL) *out_row_error = row_error;
        if (out_col_error != NULL) *out_col_error = col_error;
    }

    barrier_destroy(&shared.barrier);
    free(pthread_ids);
    free(args);
    free(shared.kernel);
    free(shared.u);
    free(shared.v);
    free(shared.thread_cost);
    free(shared.thread_row_error);
    free(shared.thread_col_error);
    return status;
}

int hungarian_run(
    const double *cost,
    int n,
    int *perm,
    double *out_cost
) {
    if (cost == NULL || n <= 0) {
        return -1;
    }

    double *u = (double *)calloc((size_t)n + 1, sizeof(double));
    double *v = (double *)calloc((size_t)n + 1, sizeof(double));
    double *minv = (double *)malloc(((size_t)n + 1) * sizeof(double));
    int *p = (int *)calloc((size_t)n + 1, sizeof(int));
    int *way = (int *)calloc((size_t)n + 1, sizeof(int));
    unsigned char *used = (unsigned char *)malloc(((size_t)n + 1) * sizeof(unsigned char));
    int *local_perm = perm;

    if (local_perm == NULL) {
        local_perm = (int *)malloc((size_t)n * sizeof(int));
    }

    if (
        u == NULL ||
        v == NULL ||
        minv == NULL ||
        p == NULL ||
        way == NULL ||
        used == NULL ||
        local_perm == NULL
    ) {
        free(u);
        free(v);
        free(minv);
        free(p);
        free(way);
        free(used);
        if (perm == NULL) free(local_perm);
        return -2;
    }

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;

        for (int j = 0; j <= n; ++j) {
            minv[j] = DBL_MAX;
            used[j] = 0;
            way[j] = 0;
        }

        while (1) {
            used[j0] = 1;
            int i0 = p[j0];
            double delta = DBL_MAX;
            int j1 = 0;

            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                double current =
                    cost[(size_t)(i0 - 1) * (size_t)n + (size_t)(j - 1)] -
                    u[i0] -
                    v[j];
                if (current < minv[j]) {
                    minv[j] = current;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }

            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if (p[j0] == 0) {
                break;
            }
        }

        while (1) {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if (j0 == 0) {
                break;
            }
        }
    }

    for (int j = 1; j <= n; ++j) {
        if (p[j] != 0) {
            local_perm[p[j] - 1] = j - 1;
        }
    }

    if (out_cost != NULL) {
        double total = 0.0;
        for (int i = 0; i < n; ++i) {
            total += cost[(size_t)i * (size_t)n + (size_t)local_perm[i]];
        }
        *out_cost = total / (double)n;
    }

    free(u);
    free(v);
    free(minv);
    free(p);
    free(way);
    free(used);
    if (perm == NULL) free(local_perm);
    return 0;
}
