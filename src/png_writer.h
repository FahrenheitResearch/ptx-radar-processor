#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

bool writePngFile(const std::filesystem::path& path,
                  const uint8_t* rgba,
                  int width,
                  int height,
                  std::string& error);
