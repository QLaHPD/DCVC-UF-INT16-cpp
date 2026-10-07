// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

namespace uf::evaluation {

int run_video_evaluation_command(int argc, char** argv);
int run_compress_time_command(int argc, char** argv);
// Private persistent benchmark worker; its control socket is descriptor 3.
int run_evaluation_worker();

}  // namespace uf::evaluation
