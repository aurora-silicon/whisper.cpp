// SPDX-License-Identifier: MIT
#include "ane/whisper-ane.h"
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

int main() {
    char temp[] = "/tmp/whisper-ane-test-XXXXXX";
    const char * dir = mkdtemp(temp);
    assert(dir);
    std::string root = dir;
    setenv("WHISPER_ANE_ENCODER", dir, 1);
    setenv("WHISPER_ANE_WORKER", ANE_TEST_WORKER, 1);
    int cpu = 0, commits = 0;
    float mel[4] = {};
    auto run = [&](whisper_ane_context * ctx) {
        return whisper_ane_encode(ctx, mel, 1, 2, 3,
            [&]() { ++cpu; return true; },
            [&](const float * result) { assert(result[0] == 7.f && result[5] == 7.f); ++commits; });
    };
    auto * ctx = whisper_ane_init("test-model");
    assert(run(ctx)); assert(cpu == 1 && commits == 0);
    whisper_ane_free(ctx);
    mkdir((root+"/package").c_str(),0700);
    std::ofstream(root+"/package/manifest.json") << "{}";
    ctx = whisper_ane_init("test-model");
    assert(run(ctx)); assert(cpu == 1 && commits == 1);
    whisper_ane_free(ctx);
    setenv("ANE_TEST_NAN","1",1);
    ctx = whisper_ane_init("test-model");
    assert(run(ctx)); assert(cpu == 2 && commits == 1);
    unsetenv("ANE_TEST_NAN");
    assert(run(ctx)); assert(cpu == 3 && commits == 1); // native stays retired
    whisper_ane_free(ctx);
    unlink((root+"/package/manifest.json").c_str());
    ctx = whisper_ane_init("test-model");
    assert(!whisper_ane_encode(ctx, mel, 1, 2, 3,
        []() { return false; }, [](const float *) { assert(false); }));
    whisper_ane_free(ctx);
    rmdir((root+"/package").c_str()); rmdir(dir);
}
