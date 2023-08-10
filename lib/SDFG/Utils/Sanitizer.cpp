// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains the sanitizer utility functions.

#include "SDFG/Utils/Sanitizer.h"

using namespace mlir;
using namespace sdfg;

/// Sanitizes the provided string to only include alphanumericals and
/// underscores.
void utils::sanitizeName(std::string &name) {
  for (unsigned i = 0; i < name.size(); ++i) {
    if (!(name[i] >= 'a' && name[i] <= 'z') &&
        !(name[i] >= 'A' && name[i] <= 'Z') &&
        !(name[i] >= '0' && name[i] <= '9')) {
      name[i] = '_';
    }
  }
}
