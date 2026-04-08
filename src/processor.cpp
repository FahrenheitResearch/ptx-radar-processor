#include "processor.h"
#include "png_writer.h"

#include "cuda/cuda_common.cuh"
#include "cuda/gpu_pipeline.cuh"
#include "cuda/preprocess.cuh"
#include "cuda/renderer.cuh"
#include "net/aws_nexrad.h"
#include "net/downloader.h"
#include "nexrad/level2_parser.h"
#include "nexrad/products.h"
#include "nexrad/stations.h"
#include "nexrad/sweep_data.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

thread_local gpu_preprocess::PreprocessWorkspace g_preprocess_workspace;

struct UtcStamp {
    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;
    int64_t epoch = 0;
};

struct WorkingSet {
    std::vector<PrecomputedSweep> sweeps;
    float station_lat = 0.0f;
    float station_lon = 0.0f;
    int total_sweeps = 0;
};

struct RunCounters {
    int listed = 0;
    int attempted = 0;
    int rendered = 0;
    int skipped = 0;
    int failed = 0;
};

std::string trimCopy(std::string text) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!text.empty() && is_space(static_cast<unsigned char>(text.front())))
        text.erase(text.begin());
    while (!text.empty() && is_space(static_cast<unsigned char>(text.back())))
        text.pop_back();
    return text;
}

std::string upperCopy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return text;
}

int64_t makeUtcEpoch(int year, int month, int day, int hour, int minute, int second) {
    std::tm tm = {};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = minute;
    tm.tm_sec = second;
#ifdef _WIN32
    return static_cast<int64_t>(_mkgmtime(&tm));
#else
    return static_cast<int64_t>(timegm(&tm));
#endif
}

bool parseUtcStamp(const std::string& text, UtcStamp& out) {
    std::string work = trimCopy(text);
    if (!work.empty() && (work.back() == 'Z' || work.back() == 'z'))
        work.pop_back();

    const int matched = std::sscanf(work.c_str(), "%d-%d-%dT%d:%d:%d",
                                    &out.year, &out.month, &out.day,
                                    &out.hour, &out.minute, &out.second);
    if (matched < 5) {
        out = {};
        const int matched_space = std::sscanf(work.c_str(), "%d-%d-%d %d:%d:%d",
                                              &out.year, &out.month, &out.day,
                                              &out.hour, &out.minute, &out.second);
        if (matched_space < 5)
            return false;
        if (matched_space == 5)
            out.second = 0;
    } else if (matched == 5) {
        out.second = 0;
    }

    out.epoch = makeUtcEpoch(out.year, out.month, out.day,
                             out.hour, out.minute, out.second);
    return true;
}

const StationInfo* findStationByCode(const std::string& station_code) {
    const std::string wanted = upperCopy(trimCopy(station_code));
    for (const auto& station : NEXRAD_STATIONS) {
        if (wanted == station.icao)
            return &station;
    }
    return nullptr;
}

int parseProductToken(const std::string& token) {
    const std::string upper = upperCopy(trimCopy(token));
    if (upper == "REF" || upper == "BR" || upper == "DBZ")
        return PROD_REF;
    if (upper == "VEL" || upper == "DV" || upper == "BV" || upper == "SRV")
        return PROD_VEL;
    if (upper == "SW")
        return PROD_SW;
    if (upper == "ZDR")
        return PROD_ZDR;
    if (upper == "CC" || upper == "RHO" || upper == "RHV")
        return PROD_CC;
    if (upper == "KDP")
        return PROD_KDP;
    if (upper == "PHI")
        return PROD_PHI;
    return -1;
}

std::string productCode(int product) {
    switch (product) {
        case PROD_REF: return "REF";
        case PROD_VEL: return "VEL";
        case PROD_SW: return "SW";
        case PROD_ZDR: return "ZDR";
        case PROD_CC: return "CC";
        case PROD_KDP: return "KDP";
        case PROD_PHI: return "PHI";
        default: return "UNK";
    }
}

std::string formatEpochLabel(int64_t epoch) {
    const std::time_t seconds = static_cast<std::time_t>(epoch);
    std::tm tm = {};
#ifdef _WIN32
    gmtime_s(&tm, &seconds);
#else
    gmtime_r(&seconds, &tm);
#endif
    std::ostringstream stream;
    stream << std::put_time(&tm, "%Y-%m-%d %H:%M:%S UTC");
    return stream.str();
}

PrecomputedSweep buildPrecomputedSweep(const ParsedSweep& sweep) {
    PrecomputedSweep pc;
    pc.meta.sweep_number = sweep.sweep_number;
    pc.elevation_angle = sweep.elevation_angle;
    pc.num_radials = static_cast<int>(sweep.radials.size());
    if (pc.num_radials == 0)
        return pc;

    pc.meta.radial_count = static_cast<uint16_t>(std::min(pc.num_radials, 0xFFFF));
    pc.meta.timing_exact = true;
    bool saw_sweep_start = false;
    bool saw_sweep_end = false;
    int64_t min_epoch = std::numeric_limits<int64_t>::max();
    int64_t max_epoch = std::numeric_limits<int64_t>::min();
    pc.azimuths.resize(pc.num_radials);
    pc.radial_time_offset_ms.resize(pc.num_radials);

    for (int radial_index = 0; radial_index < pc.num_radials; ++radial_index)
        pc.azimuths[radial_index] = sweep.radials[radial_index].azimuth;

    if (!sweep.radials.empty()) {
        pc.meta.first_azimuth_number = sweep.radials.front().azimuth_number;
        pc.meta.last_azimuth_number = sweep.radials.back().azimuth_number;
    }

    for (const auto& radial : sweep.radials) {
        if (radial.collection_epoch_ms <= 0) {
            pc.meta.timing_exact = false;
        } else {
            min_epoch = std::min(min_epoch, radial.collection_epoch_ms);
            max_epoch = std::max(max_epoch, radial.collection_epoch_ms);
        }

        switch (radial.radial_status) {
            case 0:
            case 3:
                saw_sweep_start = true;
                break;
            case 2:
            case 4:
                saw_sweep_end = true;
                break;
            default:
                break;
        }
    }

    if (min_epoch != std::numeric_limits<int64_t>::max() &&
        max_epoch != std::numeric_limits<int64_t>::min()) {
        pc.meta.sweep_start_epoch_ms = min_epoch;
        pc.meta.sweep_end_epoch_ms = max_epoch;
        pc.meta.sweep_display_epoch_ms = min_epoch + (max_epoch - min_epoch) / 2;
        for (int radial_index = 0; radial_index < pc.num_radials; ++radial_index) {
            const int64_t delta = sweep.radials[radial_index].collection_epoch_ms - min_epoch;
            pc.radial_time_offset_ms[radial_index] = static_cast<uint32_t>(std::max<int64_t>(0, delta));
        }
    } else {
        pc.radial_time_offset_ms.assign(pc.num_radials, 0);
    }
    pc.meta.boundary_complete = saw_sweep_start && saw_sweep_end;

    for (const auto& radial : sweep.radials) {
        for (const auto& moment : radial.moments) {
            const int product = moment.product_index;
            if (product < 0 || product >= NUM_PRODUCTS)
                continue;
            auto& product_data = pc.products[product];
            if (!product_data.has_data || moment.num_gates > product_data.num_gates) {
                product_data.has_data = true;
                product_data.num_gates = moment.num_gates;
                product_data.first_gate_km = moment.first_gate_m / 1000.0f;
                product_data.gate_spacing_km = moment.gate_spacing_m / 1000.0f;
                product_data.scale = moment.scale;
                product_data.offset = moment.offset;
                pc.meta.product_mask |= (1u << product);
            }
        }
    }

    for (int product = 0; product < NUM_PRODUCTS; ++product) {
        auto& product_data = pc.products[product];
        if (!product_data.has_data || product_data.num_gates <= 0)
            continue;

        const int gate_count = product_data.num_gates;
        const int radial_count = pc.num_radials;
        product_data.gates.assign(static_cast<size_t>(gate_count) * radial_count, 0);

        for (int radial_index = 0; radial_index < radial_count; ++radial_index) {
            for (const auto& moment : sweep.radials[radial_index].moments) {
                if (moment.product_index != product)
                    continue;
                const int copy_gates = std::min(static_cast<int>(moment.gates.size()), gate_count);
                for (int gate = 0; gate < copy_gates; ++gate)
                    product_data.gates[static_cast<size_t>(gate) * radial_count + radial_index] = moment.gates[gate];
                break;
            }
        }
    }

    return pc;
}

std::vector<PrecomputedSweep> buildPrecomputedSweeps(const ParsedRadarData& parsed) {
    std::vector<PrecomputedSweep> sweeps;
    sweeps.reserve(parsed.sweeps.size());
    for (const auto& sweep : parsed.sweeps)
        sweeps.push_back(buildPrecomputedSweep(sweep));
    return sweeps;
}

PrecomputedSweep buildPrecomputedSweep(const gpu_pipeline::GpuIngestResult& ingest) {
    PrecomputedSweep sweep;
    sweep.meta.radial_count = static_cast<uint16_t>(std::min(ingest.num_radials, 0xFFFF));
    sweep.elevation_angle = ingest.elevation_angle;
    sweep.num_radials = ingest.num_radials;
    if (ingest.num_radials <= 0 || !ingest.d_azimuths)
        return sweep;

    sweep.azimuths.resize(ingest.num_radials);
    sweep.radial_time_offset_ms.assign(ingest.num_radials, 0);
    CUDA_CHECK(cudaMemcpy(sweep.azimuths.data(), ingest.d_azimuths,
                          static_cast<size_t>(ingest.num_radials) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int product = 0; product < NUM_PRODUCTS; ++product) {
        if (!ingest.has_product[product] || !ingest.d_gates[product] || ingest.num_gates[product] <= 0)
            continue;

        auto& product_data = sweep.products[product];
        product_data.has_data = true;
        product_data.num_gates = ingest.num_gates[product];
        product_data.first_gate_km = ingest.first_gate_km[product];
        product_data.gate_spacing_km = ingest.gate_spacing_km[product];
        product_data.scale = ingest.scale[product];
        product_data.offset = ingest.offset[product];
        sweep.meta.product_mask |= (1u << product);
        product_data.gates.resize(static_cast<size_t>(product_data.num_gates) *
                                  static_cast<size_t>(ingest.num_radials));
        CUDA_CHECK(cudaMemcpy(product_data.gates.data(), ingest.d_gates[product],
                              product_data.gates.size() * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost));
    }

    return sweep;
}

bool normalizeGpuSweep(PrecomputedSweep& sweep) {
    if (sweep.num_radials < 10 || sweep.azimuths.size() != static_cast<size_t>(sweep.num_radials))
        return false;

    std::vector<int> keep_indices;
    keep_indices.reserve(sweep.num_radials);
    keep_indices.push_back(0);
    for (int index = 1; index < sweep.num_radials; ++index) {
        if (std::fabs(sweep.azimuths[index] - sweep.azimuths[keep_indices.back()]) < 0.01f)
            continue;
        keep_indices.push_back(index);
    }

    if (static_cast<int>(keep_indices.size()) == sweep.num_radials)
        return true;

    std::vector<float> azimuths;
    azimuths.reserve(keep_indices.size());
    for (int index : keep_indices)
        azimuths.push_back(sweep.azimuths[index]);
    sweep.azimuths.swap(azimuths);

    const int old_radials = sweep.num_radials;
    const int new_radials = static_cast<int>(keep_indices.size());
    for (int product = 0; product < NUM_PRODUCTS; ++product) {
        auto& product_data = sweep.products[product];
        if (!product_data.has_data || product_data.num_gates <= 0 || product_data.gates.empty())
            continue;

        std::vector<uint16_t> gates(static_cast<size_t>(product_data.num_gates) * new_radials);
        for (int gate = 0; gate < product_data.num_gates; ++gate) {
            for (int radial = 0; radial < new_radials; ++radial) {
                gates[static_cast<size_t>(gate) * new_radials + radial] =
                    product_data.gates[static_cast<size_t>(gate) * old_radials + keep_indices[radial]];
            }
        }
        product_data.gates.swap(gates);
    }

    sweep.num_radials = new_radials;
    return true;
}

void normalizeGpuWorkingSet(std::vector<PrecomputedSweep>& sweeps) {
    sweeps.erase(std::remove_if(sweeps.begin(), sweeps.end(),
                                [](PrecomputedSweep& sweep) { return !normalizeGpuSweep(sweep); }),
                 sweeps.end());
}

void dealiasPrecomputedSweeps(std::vector<PrecomputedSweep>& sweeps) {
    for (auto& sweep : sweeps) {
        auto& velocity = sweep.products[PROD_VEL];
        if (velocity.has_data)
            gpu_preprocess::dealiasVelocity(velocity, sweep.num_radials, &g_preprocess_workspace);
    }
}

int countProductSweeps(const std::vector<PrecomputedSweep>& sweeps, int product) {
    int count = 0;
    for (const auto& sweep : sweeps) {
        if (product >= 0 && product < NUM_PRODUCTS &&
            sweep.num_radials > 0 && sweep.products[product].has_data) {
            ++count;
        }
    }
    return count;
}

int findProductSweep(const std::vector<PrecomputedSweep>& sweeps, int product, int tilt_index) {
    int seen = 0;
    for (int index = 0; index < static_cast<int>(sweeps.size()); ++index) {
        if (product < 0 || product >= NUM_PRODUCTS)
            return -1;
        if (sweeps[index].num_radials <= 0 || !sweeps[index].products[product].has_data)
            continue;
        if (seen == tilt_index)
            return index;
        ++seen;
    }
    return -1;
}

WorkingSet buildFastLowestSweep(const std::vector<uint8_t>& decoded_bytes, bool dealias_velocity) {
    WorkingSet working_set;
    if (decoded_bytes.empty())
        return working_set;

    auto ingest = gpu_pipeline::ingestSweepGpu(decoded_bytes.data(), decoded_bytes.size());
    if (!ingest.parsed || ingest.truncated || !ingest.d_azimuths || ingest.num_radials <= 0) {
        gpu_pipeline::freeIngestResult(ingest);
        return working_set;
    }

    working_set.total_sweeps = std::max(ingest.total_sweeps, 1);
    working_set.sweeps.push_back(buildPrecomputedSweep(ingest));
    gpu_pipeline::freeIngestResult(ingest);
    normalizeGpuWorkingSet(working_set.sweeps);
    if (dealias_velocity)
        dealiasPrecomputedSweeps(working_set.sweeps);
    return working_set;
}

bool buildWorkingSet(const std::vector<uint8_t>& archive_bytes,
                     const StationInfo& station,
                     const ProcessorOptions& options,
                     WorkingSet& working_set,
                     std::string& error) {
    std::vector<uint8_t> decoded = Level2Parser::decodeArchiveBytes(archive_bytes);
    if (decoded.empty()) {
        error = "Archive decode failed";
        return false;
    }

    if (!options.cpu_only && options.tilt == 0) {
        WorkingSet fast = buildFastLowestSweep(decoded, options.dealias_velocity && options.product == PROD_VEL);
        if (!fast.sweeps.empty() && countProductSweeps(fast.sweeps, options.product) > 0) {
            fast.station_lat = station.lat;
            fast.station_lon = station.lon;
            working_set = std::move(fast);
            return true;
        }
    }

    ParsedRadarData parsed = Level2Parser::parseDecodedMessages(decoded, station.icao);
    if (parsed.sweeps.empty()) {
        error = "Parsed volume did not contain any sweeps";
        return false;
    }

    working_set.sweeps = buildPrecomputedSweeps(parsed);
    if (options.dealias_velocity && options.product == PROD_VEL)
        dealiasPrecomputedSweeps(working_set.sweeps);
    working_set.station_lat = (parsed.station_lat != 0.0f) ? parsed.station_lat : station.lat;
    working_set.station_lon = (parsed.station_lon != 0.0f) ? parsed.station_lon : station.lon;
    working_set.total_sweeps = static_cast<int>(working_set.sweeps.size());
    return true;
}

GpuStationInfo makeGpuStationInfo(const PrecomputedSweep& sweep, float station_lat, float station_lon) {
    GpuStationInfo info = {};
    info.lat = station_lat;
    info.lon = station_lon;
    info.elevation_angle = sweep.elevation_angle;
    info.num_radials = sweep.num_radials;
    for (int product = 0; product < NUM_PRODUCTS; ++product) {
        const auto& product_data = sweep.products[product];
        if (!product_data.has_data)
            continue;
        info.has_product[product] = true;
        info.num_gates[product] = product_data.num_gates;
        info.first_gate_km[product] = product_data.first_gate_km;
        info.gate_spacing_km[product] = product_data.gate_spacing_km;
        info.scale[product] = product_data.scale;
        info.offset[product] = product_data.offset;
    }
    return info;
}

std::filesystem::path outputPathForKey(const std::filesystem::path& output_dir,
                                       const std::string& key,
                                       int product,
                                       int tilt) {
    std::string stem = radarFilenameFromKey(key);
    const std::string suffix = "_" + productCode(product) + "_T" + std::to_string(tilt) + ".png";
    return output_dir / (stem + suffix);
}

std::vector<NexradFile> collectFiles(const StationInfo& station,
                                     const UtcStamp& start,
                                     const UtcStamp& end,
                                     int limit) {
    std::vector<NexradFile> matches;
    std::set<std::string> seen_keys;

    int year = start.year;
    int month = start.month;
    int day = start.day;
    while (true) {
        const std::string request = buildRadarListRequest(station, year, month, day);
        const DownloadResult list_result = Downloader::httpGet(radarDataHost(station), request);
        if (!list_result.success) {
            std::ostringstream message;
            message << "List request failed for " << station.icao << " "
                    << year << "-" << month << "-" << day << ": " << list_result.error;
            throw std::runtime_error(message.str());
        }

        const auto files = parseRadarListResponse(station, list_result.data);
        for (const auto& file : files) {
            int file_year = 0;
            int file_month = 0;
            int file_day = 0;
            int file_hour = 0;
            int file_minute = 0;
            int file_second = 0;
            if (!extractRadarFileDateTime(file.key, file_year, file_month, file_day,
                                          file_hour, file_minute, file_second)) {
                continue;
            }

            const int64_t epoch = makeUtcEpoch(file_year, file_month, file_day,
                                               file_hour, file_minute, file_second);
            if (epoch < start.epoch || epoch > end.epoch)
                continue;
            if (!seen_keys.insert(file.key).second)
                continue;

            matches.push_back(file);
            if (limit > 0 && static_cast<int>(matches.size()) >= limit)
                return matches;
        }

        if (year == end.year && month == end.month && day == end.day)
            break;
        shiftDate(year, month, day, 1);
    }

    std::sort(matches.begin(), matches.end(),
              [](const NexradFile& left, const NexradFile& right) { return left.key < right.key; });
    return matches;
}

bool uploadAndRender(const WorkingSet& working_set,
                     const ProcessorOptions& options,
                     uint32_t* d_output,
                     std::vector<uint32_t>& host_pixels,
                     std::string& error) {
    const int available_tilts = countProductSweeps(working_set.sweeps, options.product);
    if (available_tilts <= 0) {
        error = "Requested product not present in selected volume";
        return false;
    }
    if (options.tilt < 0 || options.tilt >= available_tilts) {
        std::ostringstream message;
        message << "Requested tilt " << options.tilt
                << " is unavailable for product " << productCode(options.product)
                << " (available tilts: " << available_tilts << ")";
        error = message.str();
        return false;
    }

    const int sweep_index = findProductSweep(working_set.sweeps, options.product, options.tilt);
    if (sweep_index < 0 || sweep_index >= static_cast<int>(working_set.sweeps.size())) {
        error = "Failed to resolve product/tilt sweep index";
        return false;
    }

    const auto& sweep = working_set.sweeps[sweep_index];
    GpuStationInfo info = makeGpuStationInfo(sweep, working_set.station_lat, working_set.station_lon);

    const uint16_t* gate_ptrs[NUM_PRODUCTS] = {};
    for (int product = 0; product < NUM_PRODUCTS; ++product) {
        const auto& product_data = sweep.products[product];
        if (product_data.has_data && !product_data.gates.empty())
            gate_ptrs[product] = product_data.gates.data();
    }

    gpu::allocateStation(0, info);
    gpu::uploadStationData(0, info, sweep.azimuths.data(), gate_ptrs);

    GpuViewport viewport = {};
    viewport.center_lat = static_cast<float>(options.has_center_override ? options.center_lat
                                                                         : working_set.station_lat);
    viewport.center_lon = static_cast<float>(options.has_center_override ? options.center_lon
                                                                         : working_set.station_lon);
    viewport.deg_per_pixel_x = static_cast<float>(1.0 / options.zoom);
    viewport.deg_per_pixel_y = static_cast<float>(1.0 / options.zoom);
    viewport.width = options.width;
    viewport.height = options.height;

    gpu::forwardRenderStation(viewport, 0, options.product, options.threshold, d_output);
    CUDA_CHECK(cudaMemcpy(host_pixels.data(), d_output,
                          host_pixels.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    return true;
}

std::string bytesToSize(size_t bytes) {
    const double kib = 1024.0;
    const double mib = kib * 1024.0;
    const double gib = mib * 1024.0;
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(1);
    if (bytes >= static_cast<size_t>(gib))
        stream << (bytes / gib) << " GiB";
    else if (bytes >= static_cast<size_t>(mib))
        stream << (bytes / mib) << " MiB";
    else if (bytes >= static_cast<size_t>(kib))
        stream << (bytes / kib) << " KiB";
    else
        stream << bytes << " B";
    return stream.str();
}


bool uploadAndRenderFloat(const WorkingSet& working_set,
                          const ProcessorOptions& options,
                          float* d_output_float,
                          std::vector<float>& host_floats,
                          std::string& error) {
    const int available_tilts = countProductSweeps(working_set.sweeps, options.product);
    if (available_tilts <= 0) { error = "Requested product not present"; return false; }
    if (options.tilt < 0 || options.tilt >= available_tilts) {
        error = "Requested tilt unavailable";
        return false;
    }
    const int sweep_index = findProductSweep(working_set.sweeps, options.product, options.tilt);
    if (sweep_index < 0 || sweep_index >= static_cast<int>(working_set.sweeps.size())) {
        error = "Failed to resolve sweep index";
        return false;
    }
    const auto& sweep = working_set.sweeps[sweep_index];
    GpuStationInfo info = makeGpuStationInfo(sweep, working_set.station_lat, working_set.station_lon);
    const uint16_t* gate_ptrs[NUM_PRODUCTS] = {};
    for (int product = 0; product < NUM_PRODUCTS; ++product) {
        const auto& product_data = sweep.products[product];
        if (product_data.has_data && !product_data.gates.empty())
            gate_ptrs[product] = product_data.gates.data();
    }
    gpu::allocateStation(0, info);
    gpu::uploadStationData(0, info, sweep.azimuths.data(), gate_ptrs);

    GpuViewport viewport = {};
    viewport.center_lat = static_cast<float>(options.has_center_override ? options.center_lat
                                                                         : working_set.station_lat);
    viewport.center_lon = static_cast<float>(options.has_center_override ? options.center_lon
                                                                         : working_set.station_lon);
    viewport.deg_per_pixel_x = static_cast<float>(1.0 / options.zoom);
    viewport.deg_per_pixel_y = static_cast<float>(1.0 / options.zoom);
    viewport.width = options.width;
    viewport.height = options.height;

    gpu::forwardRenderStationFloat(viewport, 0, options.product, options.threshold, d_output_float);
    CUDA_CHECK(cudaMemcpy(host_floats.data(), d_output_float,
                          host_floats.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return true;
}

} // namespace

bool parseProcessorOptions(int argc, char** argv,
                           ProcessorOptions& options,
                           bool& help_requested,
                           std::string& error) {
    help_requested = false;

    auto require_value = [&](int& index, const char* flag) -> const char* {
        if (index + 1 >= argc) {
            throw std::runtime_error(std::string("Missing value for ") + flag);
        }
        ++index;
        return argv[index];
    };

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--help" || arg == "-h") {
                help_requested = true;
                return true;
            }
            if (arg == "--station") {
                options.station = require_value(index, "--station");
            } else if (arg == "--start") {
                options.start_time_utc = require_value(index, "--start");
            } else if (arg == "--end") {
                options.end_time_utc = require_value(index, "--end");
            } else if (arg == "--out") {
                options.output_dir = require_value(index, "--out");
            } else if (arg == "--products") {
                std::string csv = require_value(index, "--products");
                std::stringstream ss(csv);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    if (tok.empty()) continue;
                    int p = parseProductToken(tok);
                    if (p < 0) { error = "Bad --products token: " + tok; return false; }
                    options.product_list.push_back(p);
                }
            } else if (arg == "--tilts") {
                std::string csv = require_value(index, "--tilts");
                std::stringstream ss(csv);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    if (!tok.empty()) options.tilt_list.push_back(std::stoi(tok));
                }
            } else if (arg == "--product") {
                const int product = parseProductToken(require_value(index, "--product"));
                if (product < 0)
                    throw std::runtime_error("Unsupported product token");
                options.product = product;
            } else if (arg == "--tilt") {
                options.tilt = std::stoi(require_value(index, "--tilt"));
            } else if (arg == "--width") {
                options.width = std::stoi(require_value(index, "--width"));
            } else if (arg == "--height") {
                options.height = std::stoi(require_value(index, "--height"));
            } else if (arg == "--zoom") {
                options.zoom = std::stod(require_value(index, "--zoom"));
            } else if (arg == "--threshold") {
                options.threshold = std::stof(require_value(index, "--threshold"));
            } else if (arg == "--center-lat") {
                options.center_lat = std::stod(require_value(index, "--center-lat"));
                options.has_center_override = true;
            } else if (arg == "--center-lon") {
                options.center_lon = std::stod(require_value(index, "--center-lon"));
                options.has_center_override = true;
            } else if (arg == "--limit") {
                options.limit = std::stoi(require_value(index, "--limit"));
            } else if (arg == "--raw-out") {
                options.raw_out = true;
            } else if (arg == "--overwrite") {
                options.overwrite = true;
            } else if (arg == "--no-dealias") {
                options.dealias_velocity = false;
            } else if (arg == "--cpu-only") {
                options.cpu_only = true;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }
    } catch (const std::exception& ex) {
        error = ex.what();
        return false;
    }

    if (options.station.empty()) {
        error = "Missing required --station";
        return false;
    }
    if (options.start_time_utc.empty()) {
        error = "Missing required --start";
        return false;
    }
    if (options.end_time_utc.empty()) {
        error = "Missing required --end";
        return false;
    }
    if (options.width <= 0 || options.height <= 0) {
        error = "Image width and height must be positive";
        return false;
    }
    if (options.zoom <= 0.0) {
        error = "Zoom must be positive";
        return false;
    }
    if (options.tilt < 0) {
        error = "Tilt must be zero or greater";
        return false;
    }
    if (options.limit < 0) {
        error = "Limit must be zero or greater";
        return false;
    }

    return true;
}

std::string processorUsage(const char* argv0) {
    std::ostringstream usage;
    usage
        << "Usage:\n"
        << "  " << argv0 << " --station KTLX --start 2025-03-30T20:00:00Z --end 2025-03-30T21:00:00Z [options]\n\n"
        << "Required:\n"
        << "  --station CODE        NEXRAD station code\n"
        << "  --start UTC           UTC start time (YYYY-MM-DDTHH:MM[:SS]Z)\n"
        << "  --end UTC             UTC end time (YYYY-MM-DDTHH:MM[:SS]Z)\n\n"
        << "Options:\n"
        << "  --out DIR             Output directory (default: output)\n"
        << "  --product TOKEN       REF|VEL|SW|ZDR|CC|KDP|PHI (default: REF)\n"
        << "  --products LIST       comma-separated product list (overrides --product)\n"
        << "  --tilts LIST          comma-separated tilt list (overrides --tilt)\n"
        << "  --tilt N              Product tilt index, zero-based (default: 0)\n"
        << "  --width N             Output width in pixels (default: 1024)\n"
        << "  --height N            Output height in pixels (default: 1024)\n"
        << "  --zoom N              Pixels per degree (default: 180)\n"
        << "  --threshold VALUE     Product threshold (default: 0)\n"
        << "  --center-lat LAT      Override viewport center latitude\n"
        << "  --center-lon LON      Override viewport center longitude\n"
        << "  --limit N             Stop after N files (default: unlimited)\n"
        << "  --raw-out             Write raw float32 .bin instead of PNG\n"
        << "  --overwrite           Re-render files even if PNG/raw already exists\n"
        << "  --no-dealias          Disable velocity dealiasing\n"
        << "  --cpu-only            Skip the fast lowest-sweep GPU ingest path\n";
    return usage.str();
}

int runProcessor(const ProcessorOptions& options) {
    UtcStamp start;
    UtcStamp end;
    if (!parseUtcStamp(options.start_time_utc, start))
        throw std::runtime_error("Could not parse --start UTC timestamp");
    if (!parseUtcStamp(options.end_time_utc, end))
        throw std::runtime_error("Could not parse --end UTC timestamp");
    if (end.epoch < start.epoch)
        throw std::runtime_error("--end must be greater than or equal to --start");

    const StationInfo* station = findStationByCode(options.station);
    if (!station)
        throw std::runtime_error("Unknown station code: " + options.station);

    std::cout << "Station: " << station->icao << " (" << station->name << ", " << station->state << ")\n";
    std::cout << "Range:   " << formatEpochLabel(start.epoch) << " -> "
              << formatEpochLabel(end.epoch) << "\n";
    std::cout << "Render:  " << options.width << "x" << options.height
              << " product=" << productCode(options.product)
              << " tilt=" << options.tilt
              << " zoom=" << options.zoom
              << " threshold=" << options.threshold << "\n";
    std::cout << "Output:  " << std::filesystem::absolute(options.output_dir).string() << "\n";

    const auto files = collectFiles(*station, start, end, options.limit);
    RunCounters counters = {};
    counters.listed = static_cast<int>(files.size());
    if (files.empty()) {
        std::cout << "No archive files matched the requested range.\n";
        return 0;
    }

    std::filesystem::create_directories(options.output_dir);
    std::cout << "Matched " << files.size() << " archive files.\n";

    bool gpu_initialized = false;
    uint32_t* d_output = nullptr;
    float* d_output_float = nullptr;
    std::vector<float> host_floats(static_cast<size_t>(options.width) * options.height);
    std::vector<uint32_t> host_pixels(static_cast<size_t>(options.width) * options.height);

    gpu::init();
    gpu_initialized = true;
    // Build effective product/tilt combo lists.
    std::vector<int> effective_products = options.product_list.empty()
        ? std::vector<int>{options.product}
        : options.product_list;
    std::vector<int> effective_tilts = options.tilt_list.empty()
        ? std::vector<int>{options.tilt}
        : options.tilt_list;
    // BATCH_LOOP_MARK

    CUDA_CHECK(cudaMalloc(&d_output, host_pixels.size() * sizeof(uint32_t)));
    if (options.raw_out) {
        CUDA_CHECK(cudaMalloc(&d_output_float, host_floats.size() * sizeof(float)));
    }
    CUDA_CHECK(cudaMemset(d_output, 0, host_pixels.size() * sizeof(uint32_t)));

    for (size_t index = 0; index < files.size(); ++index) {
        const auto& file = files[index];
        const std::filesystem::path output_path =
            outputPathForKey(options.output_dir, file.key, options.product, options.tilt);

        if (!options.overwrite && std::filesystem::exists(output_path)) {
            ++counters.skipped;
            std::cout << "[" << (index + 1) << "/" << files.size() << "] skip " << output_path.filename().string() << "\n";
            continue;
        }

        ++counters.attempted;
        std::cout << "[" << (index + 1) << "/" << files.size() << "] download " << file.key
                  << " (" << bytesToSize(file.size) << ")\n";

        const DownloadResult download = Downloader::httpGet(radarDataHost(*station), file.url);
        if (!download.success) {
            ++counters.failed;
            std::cout << "  failed: download error: " << download.error << "\n";
            continue;
        }

        WorkingSet working_set;
        std::string error;
        if (!buildWorkingSet(download.data, *station, options, working_set, error)) {
            ++counters.failed;
            std::cout << "  failed: " << error << "\n";
            continue;
        }

        if (working_set.station_lat == 0.0f && working_set.station_lon == 0.0f) {
            working_set.station_lat = station->lat;
            working_set.station_lon = station->lon;
        }

        bool any_failed = false;
        bool any_rendered = false;
        for (int eff_prod : effective_products) {
            for (int eff_tilt : effective_tilts) {
                ProcessorOptions per_combo = options;
                per_combo.product = eff_prod;
                per_combo.tilt = eff_tilt;
                std::filesystem::path combo_path =
                    outputPathForKey(options.output_dir, file.key, eff_prod, eff_tilt);
                if (options.raw_out) {
                    combo_path.replace_extension(".bin");
                }
                if (!options.overwrite && std::filesystem::exists(combo_path)) {
                    continue;
                }
                if (options.raw_out) {
                    std::string err2;
                    if (!uploadAndRenderFloat(working_set, per_combo, d_output_float, host_floats, err2)) {
                        any_failed = true;
                        std::cout << "  failed " << productCode(eff_prod) << " T" << eff_tilt << ": " << err2 << "\n";
                        continue;
                    }
                    FILE* fp = fopen(combo_path.string().c_str(), "wb");
                    if (!fp) { any_failed = true; std::cout << "  failed bin open\n"; continue; }
                    fwrite(host_floats.data(), sizeof(float), host_floats.size(), fp);
                    fclose(fp);
                    any_rendered = true;
                } else {
                    std::string err2;
                    if (!uploadAndRender(working_set, per_combo, d_output, host_pixels, err2)) {
                        any_failed = true;
                        continue;
                    }
                    if (!writePngFile(combo_path,
                                      reinterpret_cast<const uint8_t*>(host_pixels.data()),
                                      options.width, options.height, err2)) {
                        any_failed = true;
                        continue;
                    }
                    any_rendered = true;
                }
            }
        }
        if (any_rendered) {
            ++counters.rendered;
            std::cout << "  wrote " << effective_products.size() * effective_tilts.size()
                      << " channels for " << std::filesystem::path(file.key).filename().string() << "\n";
        } else if (any_failed) {
            ++counters.failed;
        } else {
            ++counters.skipped;
        }
    }

    if (d_output_float) { cudaFree(d_output_float); d_output_float = nullptr; }
    if (d_output) {
        cudaFree(d_output);
        d_output = nullptr;
    }
    if (gpu_initialized)
        gpu::shutdown();

    std::cout << "\nSummary: "
              << "matched=" << counters.listed
              << " attempted=" << counters.attempted
              << " rendered=" << counters.rendered
              << " skipped=" << counters.skipped
              << " failed=" << counters.failed
              << "\n";

    return counters.rendered > 0 ? 0 : (counters.failed == 0 ? 0 : 1);
}
