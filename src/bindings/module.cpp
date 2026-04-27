#include <pybind11/pybind11.h>

#include <string>

#include "core/video_processor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(video_processor, m) {
    m.doc() = "CUDA video processor module for UYVY deinterlace/crop/zoom placeholder SR";

    py::class_<vp::VideoProcessor>(m, "VideoProcessor")
        .def(
            py::init<int, int, int, int, int, int, bool, int>(),
            py::arg("width") = 1920,
            py::arg("height") = 1080,
            py::arg("roi_x") = 0,
            py::arg("roi_y") = 0,
            py::arg("roi_w") = 1920,
            py::arg("roi_h") = 1080,
            py::arg("enable_placeholder_sr") = true,
            py::arg("sr_scale") = 0
        )
        .def(
            "process_frame",
            [](vp::VideoProcessor& self, py::bytes frame) {
                const std::string input = static_cast<std::string>(frame);
                const std::string output = self.ProcessFrame(input);
                return py::bytes(output);
            },
            py::arg("frame"),
            "Process one 1920x1080 interlaced UYVY frame and return UYVY bytes."
        )
        .def_property_readonly("width", &vp::VideoProcessor::width)
        .def_property_readonly("height", &vp::VideoProcessor::height)
        .def_property_readonly("sr_scale", &vp::VideoProcessor::sr_scale);
}
