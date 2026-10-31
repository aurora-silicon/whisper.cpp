// SPDX-License-Identifier: MIT
#pragma once
#include <functional>
struct whisper_ane_context;
whisper_ane_context * whisper_ane_init(const char * model_path);
void whisper_ane_free(whisper_ane_context * ctx);
// CPU recomputes both convolution and encoder from the original mel input.
bool whisper_ane_encode(whisper_ane_context * ctx, const float * mel,
                        int n_mels, int n_ctx, int n_state,
                        const std::function<bool()> & cpu,
                        const std::function<void(const float *)> & commit);
