#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtx_vsr_engine.h"

namespace py = pybind11;

namespace {

void ValidateInputFrame(const py::array& frame, int expected_h, int expected_w) {
    if (!py::isinstance<py::array_t<std::uint8_t>>(frame)) {
        throw py::value_error("Input frame must have dtype=uint8");
    }

    if (frame.ndim() != 3) {
        throw py::value_error("Input frame must have shape [H, W, 4]");
    }

    if (frame.shape(2) != 4) {
        throw py::value_error("Input frame channel dimension must be 4 (RGBA)");
    }

    if (frame.shape(0) != expected_h || frame.shape(1) != expected_w) {
        throw py::value_error("Input frame shape does not match constructor input_width/input_height");
    }

    if (!(frame.flags() & py::array::c_style)) {
        throw py::value_error("Input frame must be C-contiguous");
    }
}

}  // namespace

PYBIND11_MODULE(rtx_vsr, m) {
    m.doc() = "Windows D3D11 RTX Video SDK VSR proof-of-concept wrapper";

    py::class_<rtx_vsr::RTXVideoSREngine>(m, "RTXVideoSR")
           .def(py::init<int, int, int, int, const std::string&, bool, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>(),
             py::arg("input_width"),
             py::arg("input_height"),
             py::arg("output_width"),
             py::arg("output_height"),
               py::arg("quality") = "high",
               py::arg("thdr_enabled") = false,
               py::arg("thdr_contrast") = 50,
               py::arg("thdr_saturation") = 50,
               py::arg("thdr_middle_gray") = 50,
               py::arg("thdr_max_luminance") = 1000)
        .def("process_rgba",
             [](rtx_vsr::RTXVideoSREngine& self, const py::array& frame) {
                 ValidateInputFrame(frame, self.input_height(), self.input_width());

                 auto input = py::array_t<std::uint8_t, py::array::c_style>::ensure(frame);
                 if (!input) {
                     throw py::value_error("Failed to read input frame as C-contiguous uint8 array");
                 }

                 const std::size_t input_bytes = static_cast<std::size_t>(self.input_width()) * static_cast<std::size_t>(self.input_height()) * 4;
                 std::vector<std::uint8_t> out = self.ProcessRGBA(input.data(), input_bytes);

                 py::array_t<std::uint8_t> result({self.output_height(), self.output_width(), 4});
                 auto result_buf = result.request();
                 if (result_buf.size != static_cast<py::ssize_t>(out.size())) {
                     throw std::runtime_error("Output size mismatch");
                 }

                 std::memcpy(result.mutable_data(), out.data(), out.size());
                 return result;
             },
             py::arg("frame"),
             "Process one RGBA frame and return enhanced RGBA output")
        .def("close", &rtx_vsr::RTXVideoSREngine::Close,
             "Release D3D11 and RTX SDK resources")
        .def(
            "process_cuda_ptr",
            [](rtx_vsr::RTXVideoSREngine&, std::uint64_t, std::uint64_t, std::uint64_t) {
                throw std::runtime_error(
                    "TODO: implement zero-copy path via CUDA pointer or D3D interop in a later milestone");
            },
            py::arg("input_ptr"),
            py::arg("output_ptr"),
            py::arg("stream_ptr"),
            "TODO placeholder for future zero-copy integration")
        .def_property_readonly("input_width", &rtx_vsr::RTXVideoSREngine::input_width)
        .def_property_readonly("input_height", &rtx_vsr::RTXVideoSREngine::input_height)
        .def_property_readonly("output_width", &rtx_vsr::RTXVideoSREngine::output_width)
        .def_property_readonly("output_height", &rtx_vsr::RTXVideoSREngine::output_height)
        .def_property_readonly("quality_level", [](const rtx_vsr::RTXVideoSREngine& self) {
            return self.settings().quality_level;
        })
        .def_property_readonly("thdr_enabled", [](const rtx_vsr::RTXVideoSREngine& self) {
            return self.settings().thdr_enabled;
        })
        .def_property_readonly("thdr_contrast", [](const rtx_vsr::RTXVideoSREngine& self) {
            return self.settings().thdr_contrast;
        })
        .def_property_readonly("thdr_saturation", [](const rtx_vsr::RTXVideoSREngine& self) {
            return self.settings().thdr_saturation;
        })
        .def_property_readonly("thdr_middle_gray", [](const rtx_vsr::RTXVideoSREngine& self) {
            return self.settings().thdr_middle_gray;
        })
        .def_property_readonly("thdr_max_luminance", [](const rtx_vsr::RTXVideoSREngine& self) {
            return self.settings().thdr_max_luminance;
        });
}
