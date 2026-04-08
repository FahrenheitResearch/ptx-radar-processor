#include "png_writer.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wincodec.h>
#endif

#include <cstdio>
#include <filesystem>
#include <vector>

namespace {

#ifdef _WIN32
template <typename T>
void releaseCom(T*& ptr) {
    if (ptr) {
        ptr->Release();
        ptr = nullptr;
    }
}

std::string hresultMessage(HRESULT hr) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "HRESULT 0x%08X", static_cast<unsigned>(hr));
    return buffer;
}
#endif

} // namespace

bool writePngFile(const std::filesystem::path& path,
                  const uint8_t* rgba,
                  int width,
                  int height,
                  std::string& error) {
#ifndef _WIN32
    (void)path;
    (void)rgba;
    (void)width;
    (void)height;
    error = "PNG writing is only implemented on Windows in this extraction";
    return false;
#else
    if (!rgba || width <= 0 || height <= 0) {
        error = "Invalid PNG buffer dimensions";
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        error = "Failed to create output directory: " + ec.message();
        return false;
    }

    HRESULT init_hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool should_uninit = SUCCEEDED(init_hr);
    if (FAILED(init_hr) && init_hr != RPC_E_CHANGED_MODE) {
        error = "CoInitializeEx failed: " + hresultMessage(init_hr);
        return false;
    }

    std::vector<uint8_t> bgra(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
    for (size_t i = 0; i < bgra.size(); i += 4) {
        bgra[i + 0] = rgba[i + 2];
        bgra[i + 1] = rgba[i + 1];
        bgra[i + 2] = rgba[i + 0];
        bgra[i + 3] = rgba[i + 3];
    }

    IWICImagingFactory* factory = nullptr;
    IWICStream* stream = nullptr;
    IWICBitmapEncoder* encoder = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    IPropertyBag2* props = nullptr;

    auto cleanup = [&]() {
        releaseCom(props);
        releaseCom(frame);
        releaseCom(encoder);
        releaseCom(stream);
        releaseCom(factory);
        if (should_uninit)
            CoUninitialize();
    };

    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        error = "CoCreateInstance(CLSID_WICImagingFactory) failed: " + hresultMessage(hr);
        cleanup();
        return false;
    }

    hr = factory->CreateStream(&stream);
    if (FAILED(hr)) {
        error = "CreateStream failed: " + hresultMessage(hr);
        cleanup();
        return false;
    }

    hr = stream->InitializeFromFilename(path.wstring().c_str(), GENERIC_WRITE);
    if (FAILED(hr)) {
        error = "InitializeFromFilename failed: " + hresultMessage(hr);
        cleanup();
        return false;
    }

    hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder);
    if (FAILED(hr)) {
        error = "CreateEncoder failed: " + hresultMessage(hr);
        cleanup();
        return false;
    }

    hr = encoder->Initialize(stream, WICBitmapEncoderNoCache);
    if (FAILED(hr)) {
        error = "Encoder Initialize failed: " + hresultMessage(hr);
        cleanup();
        return false;
    }

    hr = encoder->CreateNewFrame(&frame, &props);
    if (FAILED(hr)) {
        error = "CreateNewFrame failed: " + hresultMessage(hr);
        cleanup();
        return false;
    }

    hr = frame->Initialize(props);
    if (FAILED(hr)) {
        error = "Frame Initialize failed: " + hresultMessage(hr);
        cleanup();
        return false;
    }

    hr = frame->SetSize(static_cast<UINT>(width), static_cast<UINT>(height));
    if (FAILED(hr)) {
        error = "SetSize failed: " + hresultMessage(hr);
        cleanup();
        return false;
    }

    WICPixelFormatGUID pixel_format = GUID_WICPixelFormat32bppBGRA;
    hr = frame->SetPixelFormat(&pixel_format);
    if (FAILED(hr)) {
        error = "SetPixelFormat failed: " + hresultMessage(hr);
        cleanup();
        return false;
    }
    if (pixel_format != GUID_WICPixelFormat32bppBGRA) {
        error = "Unexpected WIC pixel format negotiation";
        cleanup();
        return false;
    }

    const UINT stride = static_cast<UINT>(width * 4);
    const UINT image_size = static_cast<UINT>(bgra.size());
    hr = frame->WritePixels(static_cast<UINT>(height), stride, image_size, bgra.data());
    if (FAILED(hr)) {
        error = "WritePixels failed: " + hresultMessage(hr);
        cleanup();
        return false;
    }

    hr = frame->Commit();
    if (FAILED(hr)) {
        error = "Frame Commit failed: " + hresultMessage(hr);
        cleanup();
        return false;
    }

    hr = encoder->Commit();
    if (FAILED(hr)) {
        error = "Encoder Commit failed: " + hresultMessage(hr);
        cleanup();
        return false;
    }

    cleanup();
    return true;
#endif
}
