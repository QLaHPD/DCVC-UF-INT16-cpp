// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#pragma once
#include "uf/evaluation_json.hpp"
namespace uf::tcm {
uf::evaluation::Json evaluate_document(const uf::evaluation::Json& point, bool use_cuda, int worker_index);
}  // namespace uf::tcm
