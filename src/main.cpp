#include "processor.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    ProcessorOptions options;
    bool help_requested = false;
    std::string error;

    if (!parseProcessorOptions(argc, argv, options, help_requested, error)) {
        std::cerr << error << "\n\n" << processorUsage(argv[0]) << "\n";
        return 1;
    }

    if (help_requested) {
        std::cout << processorUsage(argv[0]) << "\n";
        return 0;
    }

    try {
        return runProcessor(options);
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Fatal error: unknown exception\n";
        return 1;
    }
}
