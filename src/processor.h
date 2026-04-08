#pragma once

#include <filesystem>
#include <string>

struct ProcessorOptions {
    std::string station;
    std::string start_time_utc;
    std::string end_time_utc;
    std::filesystem::path output_dir = "output";
    int width = 1024;
    int height = 1024;
    double zoom = 180.0;
    int product = 0;
    int tilt = 0;
    float threshold = 0.0f;
    bool overwrite = false;
    bool dealias_velocity = true;
    bool cpu_only = false;
    bool has_center_override = false;
    double center_lat = 0.0;
    double center_lon = 0.0;
    int limit = 0;
};

bool parseProcessorOptions(int argc, char** argv,
                           ProcessorOptions& options,
                           bool& help_requested,
                           std::string& error);

std::string processorUsage(const char* argv0);

int runProcessor(const ProcessorOptions& options);
