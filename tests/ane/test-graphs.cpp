// SPDX-License-Identifier: MIT
// Exercise the real scheduler handoff twice with a fake constant-output worker.
// This is an allocation/lifecycle test, not numerical inference qualification.
#include "whisper.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
int main() {
    char temp[] = "/tmp/whisper-ane-graphs-XXXXXX";
    const char * dir = mkdtemp(temp);
    assert(dir);
    const std::string root = dir, prefix = root+"/encoder";
    mkdir((root+"/package").c_str(),0700);
    std::ofstream(root+"/package/manifest.json") << "{}";
    setenv("WHISPER_ANE_ENCODER",dir,1);
    setenv("WHISPER_ANE_WORKER",ANE_TEST_WORKER,1);
    setenv("WHISPER_ENCODER_DUMP",prefix.c_str(),1);
    unsetenv("ANE_TEST_NAN");
    auto params = whisper_context_default_params();
    params.use_gpu = false;
    params.flash_attn = false;
    auto * ctx = whisper_init_from_file_with_params(ANE_TEST_MODEL, params);
    assert(ctx);
    std::vector<float> mel(80*3000,0.f);
    assert(whisper_set_mel(ctx,mel.data(),3000,80) == 0);
    for (int i = 0; i < 2; ++i) {
        assert(whisper_encode(ctx,0,2) == 0);
        const std::string file = prefix+"-"+std::to_string(i)+".f32";
        FILE * f = fopen(file.c_str(),"rb");
        assert(f);
        float value;
        size_t count = 0;
        while (fread(&value,sizeof(value),1,f) == 1) { assert(value == 7.f); ++count; }
        fclose(f);
        assert(count == size_t(1500*384));
        unlink(file.c_str());
    }
    whisper_free(ctx);
    unlink((root+"/package/manifest.json").c_str());
    rmdir((root+"/package").c_str()); rmdir(dir);
}
