#include <pybind11/pybind11.h>

#include <string>

#include "core/video_processor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(video_processor, m) {
    m.doc() = "CUDA video processor module for UYVY deinterlace/crop/zoom basic scaling";

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
        .def(
            "process_frame_no_deinterlace",
            [](vp::VideoProcessor& self, py::bytes frame) {
                const std::string input = static_cast<std::string>(frame);
                const std::string output = self.ProcessFrameNoDeinterlace(input);
                return py::bytes(output);
            },
            py::arg("frame"),
            "Process one UYVY frame while skipping Bob deinterlacing in this pass."
        )
        .def(
            "process_frame_deinterlace_only",
            [](vp::VideoProcessor& self, py::bytes frame) {
                const std::string input = static_cast<std::string>(frame);
                const std::string output = self.ProcessFrameDeinterlaceOnly(input);
                return py::bytes(output);
            },
            py::arg("frame"),
            "Apply Bob deinterlacing and return deinterlaced UYVY bytes without ROI scaling."
        )
        .def(
            "set_roi",
            &vp::VideoProcessor::SetRoi,
            py::arg("roi_x"),
            py::arg("roi_y"),
            py::arg("roi_w"),
            py::arg("roi_h"),
            "Set ROI rectangle; values are clamped to valid frame bounds."
        )
        .def(
            "set_roi_position",
            &vp::VideoProcessor::SetRoiPosition,
            py::arg("roi_x"),
            py::arg("roi_y"),
            "Set ROI position; size is preserved and full ROI is clamped."
        )
        .def(
            "set_roi_size",
            &vp::VideoProcessor::SetRoiSize,
            py::arg("roi_w"),
            py::arg("roi_h"),
            "Set ROI size; position is preserved and full ROI is clamped."
        )
        .def(
            "get_roi",
            [](const vp::VideoProcessor& self) {
                int roi_x = 0;
                int roi_y = 0;
                int roi_w = 0;
                int roi_h = 0;
                self.GetRoi(roi_x, roi_y, roi_w, roi_h);
                return py::make_tuple(roi_x, roi_y, roi_w, roi_h);
            },
            "Get current ROI as (roi_x, roi_y, roi_w, roi_h)."
        )
        .def("set_sr_mode_auto", &vp::VideoProcessor::SetSrModeAuto, "Enable auto SR scale mode.")
        .def("set_basic_scaling_mode_auto", &vp::VideoProcessor::SetSrModeAuto, "Enable auto basic-scaling mode.")
        .def(
            "set_sr_scale_manual",
            &vp::VideoProcessor::SetSrScaleManual,
            py::arg("sr_scale"),
            "Set manual SR scale to one of [2, 4, 8, 16]; may fall back on low memory."
        )
        .def(
            "set_basic_scaling_manual",
            &vp::VideoProcessor::SetSrScaleManual,
            py::arg("scale"),
            "Set manual basic scaling to one of [2, 4, 8, 16]; may fall back on low memory."
        )
        .def(
            "get_effective_sr_scale",
            &vp::VideoProcessor::GetEffectiveSrScale,
            "Get the currently active SR scale after any fallback."
        )
        .def(
            "get_effective_basic_scaling",
            &vp::VideoProcessor::GetEffectiveSrScale,
            "Get the currently active basic scaling after any fallback."
        )
        .def(
            "set_max_auto_sr_scale",
            &vp::VideoProcessor::SetMaxAutoSrScale,
            py::arg("sr_scale"),
            "Set the maximum allowed auto SR scale to one of [2, 4, 8, 16]."
        )
        .def(
            "set_max_auto_basic_scaling",
            &vp::VideoProcessor::SetMaxAutoSrScale,
            py::arg("scale"),
            "Set the maximum allowed auto basic scaling to one of [2, 4, 8, 16]."
        )
        .def(
            "get_max_auto_sr_scale",
            &vp::VideoProcessor::GetMaxAutoSrScale,
            "Get the configured maximum auto SR scale."
        )
        .def(
            "get_max_auto_basic_scaling",
            &vp::VideoProcessor::GetMaxAutoSrScale,
            "Get the configured maximum auto basic scaling."
        )
        .def(
            "set_sr_flavor",
            &vp::VideoProcessor::SetSrFlavorByName,
            py::arg("sr_flavor"),
            "Set SR flavor to one of [bilinear, bicubic, bicubic_sharpen]."
        )
        .def(
            "set_basic_scaling_method",
            &vp::VideoProcessor::SetSrFlavorByName,
            py::arg("method"),
            "Set basic scaling method to one of [bilinear, bicubic, bicubic_sharpen]."
        )
        .def(
            "get_sr_flavor",
            &vp::VideoProcessor::GetSrFlavorName,
            "Get current SR flavor name."
        )
        .def(
            "get_basic_scaling_method",
            &vp::VideoProcessor::GetSrFlavorName,
            "Get current basic scaling method name."
        )
        .def(
            "set_deinterlace_enabled",
            &vp::VideoProcessor::SetDeinterlaceEnabled,
            py::arg("enabled"),
            "Enable or disable Bob deinterlacing before ROI/crop processing."
        )
        .def(
            "is_deinterlace_enabled",
            &vp::VideoProcessor::IsDeinterlaceEnabled,
            "Return whether Bob deinterlacing is currently enabled."
        )
        .def_property_readonly("width", &vp::VideoProcessor::width)
        .def_property_readonly("height", &vp::VideoProcessor::height)
        .def_property_readonly("sr_scale", &vp::VideoProcessor::sr_scale)
        .def_property_readonly("sr_auto_mode", &vp::VideoProcessor::IsSrAutoMode)
        .def_property_readonly("effective_sr_scale", &vp::VideoProcessor::GetEffectiveSrScale)
        .def_property_readonly("basic_scaling", &vp::VideoProcessor::GetEffectiveSrScale)
        .def_property("sr_flavor", &vp::VideoProcessor::GetSrFlavorName, &vp::VideoProcessor::SetSrFlavorByName)
        .def_property("basic_scaling_method", &vp::VideoProcessor::GetSrFlavorName, &vp::VideoProcessor::SetSrFlavorByName)
        .def_property("max_auto_sr_scale", &vp::VideoProcessor::GetMaxAutoSrScale, &vp::VideoProcessor::SetMaxAutoSrScale)
        .def_property("max_auto_basic_scaling", &vp::VideoProcessor::GetMaxAutoSrScale, &vp::VideoProcessor::SetMaxAutoSrScale)
        .def_property("deinterlace_enabled", &vp::VideoProcessor::IsDeinterlaceEnabled, &vp::VideoProcessor::SetDeinterlaceEnabled);
}
