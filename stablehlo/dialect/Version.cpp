/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "stablehlo/dialect/Version.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace vhlo {
namespace {

struct VersionTagDate {
  Version ver;
  std::tm tagDate;
};

static std::tm makeTm(int year, int mon, int day) {
  std::tm t;
  t.tm_year = year - 1900;  // Year since 1900
  t.tm_mon = mon - 1;       // Month: December (0-based, so 11 is December)
  t.tm_mday = day;
  t.tm_hour = 0;
  t.tm_min = 0;
  t.tm_sec = 0;
  return t;
}

// Tags: https://github.com/openxla/stablehlo/tags
static std::array<VersionTagDate, 8> releases = {
    VersionTagDate{Version(1, 7, 0), makeTm(2024, 9, 5)},
    VersionTagDate{Version(1, 6, 0), makeTm(2024, 8, 16)},
    VersionTagDate{Version(1, 5, 0), makeTm(2024, 8, 1)},
    VersionTagDate{Version(1, 4, 0), makeTm(2024, 7, 16)},
    VersionTagDate{Version(1, 3, 0), makeTm(2024, 7, 16)},
    VersionTagDate{Version(1, 2, 0), makeTm(2024, 7, 8)},
    VersionTagDate{Version(1, 1, 0), makeTm(2024, 5, 30)},
    VersionTagDate{Version(1, 0, 0), makeTm(2024, 5, 14)},
};

static int daysSince(const std::time_t pastTime) {
  // Get the current UTC time
  auto now = std::chrono::system_clock::now();
  std::time_t nowUtcTime = std::chrono::system_clock::to_time_t(now);
  double diffSec = std::difftime(nowUtcTime, pastTime);
  return diffSec / 86400;
}

static Version getVersionTaggedOnOrAfterDays(int daysAgo) {
  for (auto& rel : releases) {
    time_t tagTime = std::mktime(&(rel.tagDate));
    int diff = daysSince(tagTime);
    if (diff >= daysAgo) {
      return rel.ver;
    }
  }
  return Version::getMinimumVersion();
}

// Helper function for number to string.
// Precondition that numRef is a valid decimal digit.
static int64_t parseNumber(llvm::StringRef numRef) {
  int64_t num;
  if (numRef.getAsInteger(/*radix=*/10, num)) {
    llvm::report_fatal_error("failed to parse version number");
  }
  return num;
}

/// Validate version argument is `#.#.#` (ex: 0.9.0, 0.99.0, 1.2.3)
/// Returns the vector of 3 matches (major, minor, patch) if successful,
/// else returns failure.
static FailureOr<std::array<int64_t, 3>> extractVersionNumbers(
    llvm::StringRef versionRef) {
  llvm::Regex versionRegex("^([0-9]+)\\.([0-9]+)\\.([0-9]+)$");
  llvm::SmallVector<llvm::StringRef> matches;
  if (!versionRegex.match(versionRef, &matches)) return failure();
  return std::array<int64_t, 3>{parseNumber(matches[1]),
                                parseNumber(matches[2]),
                                parseNumber(matches[3])};
}

}  // namespace

FailureOr<Version> Version::fromString(llvm::StringRef versionRef) {
  auto failOrVersionArray = extractVersionNumbers(versionRef);
  if (failed(failOrVersionArray)) return failure();
  auto versionArr = *failOrVersionArray;
  return Version(versionArr[0], versionArr[1], versionArr[2]);
}

FailureOr<int64_t> Version::getBytecodeVersion() const {
  if (*this < Version(0, 9, 0)) return failure();
  if (*this < Version(0, 10, 0)) return 0;
  if (*this < Version(0, 12, 0)) return 1;
  if (*this < Version(0, 14, 0)) return 3;
  if (*this < Version(0, 15, 0)) return 4;  // (revised from 5 to 4 in #1827)
  if (*this <= getCurrentVersion()) return 6;
  return failure();
}

Version Version::fromCompatibilityRequirement(
    CompatibilityRequirement requirement) {
  // Compatibility requirement versions can be updated as needed, as long as the
  // version satisifies the requirement.
  // The time frames used are from the date that the release was tagged on, not
  // merged. The tag date is when the version has been verified and exported to
  // XLA. See: https://github.com/openxla/stablehlo/tags
  switch (requirement) {
    case CompatibilityRequirement::NONE:
      return Version::getCurrentVersion();
    case CompatibilityRequirement::WEEK_4:
      return getVersionTaggedOnOrAfterDays(28);
    case CompatibilityRequirement::WEEK_12:
      return getVersionTaggedOnOrAfterDays(84);
    case CompatibilityRequirement::MAX:
      return Version::getMinimumVersion();
  }
  llvm::report_fatal_error("Unhandled compatibility requirement");
}

mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const Version& version) {
  return diag << version.toString();
}
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Version& version) {
  return os << version.toString();
}

}  // namespace vhlo
}  // namespace mlir
