// SPDX-License-Identifier: MIT
#include "whisper-ane.h"
#include <ane.h>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <spawn.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;

struct whisper_ane_context {
    ane_fallback_state policy = {};
    std::string package, model, worker, python;
};

whisper_ane_context * whisper_ane_init(const char * model_path) {
    const char * package = getenv("WHISPER_ANE_ENCODER");
    if (!package || !*package) {
        return nullptr;
    }
    auto * ctx = new whisper_ane_context;
    ctx->package = package;
    ctx->model = model_path;
    const char * worker = getenv("WHISPER_ANE_WORKER");
    const char * python = getenv("WHISPER_ANE_PYTHON");
    ctx->worker = worker ? worker : WHISPER_ANE_WORKER_PATH;
    ctx->python = python ? python : "python3";
    // Missing/refused packages must never touch a device.
    const bool available = access((ctx->package + "/package/manifest.json").c_str(), R_OK) == 0;
    ane_fallback_init(&ctx->policy, available);
    if (!available) {
        fprintf(stderr, "ane: no compiled encoder package; using CPU\n");
    }
    return ctx;
}

void whisper_ane_free(whisper_ane_context * ctx) {
    delete ctx;
}

static bool write_floats(const std::string & path, const float * data, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    const bool ok = fwrite(data, sizeof(float), n, f) == n;
    return fclose(f) == 0 && ok;
}

static int native_encode(whisper_ane_context & ctx, const float * mel,
                         int n_mels, int n_ctx, int n_state,
                         const std::function<void(const float *)> & commit) {
    char pattern[] = "/tmp/whisper-ane-XXXXXX";
    const char * dir = mkdtemp(pattern);
    if (!dir) {
        return -errno;
    }
    const std::string root = dir, input = root + "/mel.f32", output = root + "/encoder.f32";
    int rc = -EIO;
    if (write_floats(input, mel, size_t(n_mels)*2*n_ctx)) {
        std::vector<std::string> args = {ctx.python, ctx.worker, "--bundle", ctx.package,
            "--ggml", ctx.model, "--input", input, "--output", output,
            "--n-mels", std::to_string(n_mels), "--n-ctx", std::to_string(n_ctx),
            "--n-state", std::to_string(n_state)};
        std::vector<char *> argv;
        for (auto & arg : args) {
            argv.push_back(&arg[0]);
        }
        argv.push_back(nullptr);
        pid_t pid;
        posix_spawnattr_t attr;
        int error = posix_spawnattr_init(&attr);
        if (error) {
            fprintf(stderr, "ane: spawn setup failed; preserved %s\n", dir);
            return -error;
        }
        sigset_t mask;
        sigemptyset(&mask);
        sigaddset(&mask, SIGINT);
        error = posix_spawnattr_setsigmask(&attr, &mask);
        if (!error) { error = posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSIGMASK); }
        if (!error) { error = posix_spawnp(&pid, argv[0], nullptr, &attr, argv.data(), environ); }
        posix_spawnattr_destroy(&attr);
        if (!error) {
            int status = 0;
            pid_t waited;
            // The compiler runner owns in-flight ioctls and cleanup. Do not kill
            // it mid-submission or treat interrupted waitpid as a CPU retry.
            do { waited = waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
            if (waited == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                std::vector<float> result(size_t(n_ctx)*n_state);
                FILE * f = fopen(output.c_str(), "rb");
                if (f) {
                    const bool complete = fread(result.data(), sizeof(float), result.size(), f) == result.size();
                    const bool eof = fgetc(f) == EOF;
                    fclose(f);
                    bool finite = true;
                    for (float value : result) { finite = finite && std::isfinite(value); }
                    if (complete && eof && finite) {
                        commit(result.data());
                        rc = 0;
                    }
                }
            }
        } else {
            rc = -error;
        }
    }
    if (rc) {
        fprintf(stderr, "ane: native encoder failed; preserved evidence in %s; using CPU\n", dir);
    } else if (getenv("WHISPER_ANE_KEEP_RECEIPTS")) {
        fprintf(stderr, "ane: native receipt preserved in %s\n", dir);
    } else {
        unlink(input.c_str());
        unlink(output.c_str());
        unlink((root + "/encoder.fp16").c_str());
        unlink((root + "/mel.fp16").c_str());
        unlink((root + "/runner.log").c_str());
        unlink((root + "/command.json").c_str());
        rmdir(dir);
    }
    return rc;
}

bool whisper_ane_encode(whisper_ane_context * ctx, const float * mel,
                        int n_mels, int n_ctx, int n_state,
                        const std::function<bool()> & cpu,
                        const std::function<void(const float *)> & commit) {
    struct invocation {
        std::function<int()> native, cpu;
    } work = {
        [&]() { return native_encode(*ctx, mel, n_mels, n_ctx, n_state, commit); },
        [&]() { return cpu() ? 0 : -EIO; }
    };
    ane_fallback_call call = {};
    int rc = ane_fallback_run(&ctx->policy,
        [](void * p) { return static_cast<invocation *>(p)->native(); },
        [](void * p) { return static_cast<invocation *>(p)->cpu(); }, &work, &call);
    fprintf(stderr, "ane: encoder backend=%s reason=%u native_errno=%d elapsed_ms=%.3f\n",
            call.backend == ANE_INFER_NATIVE ? "ane" : "cpu", call.reason,
            call.native_error, call.latency_ns/1e6);
    return rc == 0;
}
