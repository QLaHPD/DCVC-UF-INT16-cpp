// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Layout/text/legend/drawing adapt Matplotlib 3.10.9; notice: licenses/README.md.
#include "rd_plot.hpp"
#include "rd_axis.hpp"
#include "rd_layout.hpp"
#include "uf/process.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#ifdef UF_HAS_CAIRO
#include <cairo.h>
#endif

namespace uf::evaluation {
#if defined(UF_HAS_CAIRO) && CAIRO_HAS_PNG_FUNCTIONS
namespace {

using Surface = std::unique_ptr<cairo_surface_t, decltype(&cairo_surface_destroy)>;
using Context = std::unique_ptr<cairo_t, decltype(&cairo_destroy)>;
constexpr std::array<std::array<int, 3>, 10> colors{{
    {31,119,180}, {255,127,14}, {44,160,44}, {214,39,40}, {148,103,189},
    {140,86,75}, {227,119,194}, {127,127,127}, {188,189,34}, {23,190,207}}};
// Coordinates are at 100 dpi, scaled by three for the 300 dpi PNG. Matplotlib
// expresses font and stroke sizes in points (1/72 inch), not these pixels.
constexpr double point_unit = 100.0 / 72.0;
constexpr double label_size = 10 * point_unit;
constexpr double line_width = 1.5 * point_unit;

void check(cairo_status_t status) {
    if (status != CAIRO_STATUS_SUCCESS) throw std::runtime_error(cairo_status_to_string(status));
}

using Box = plot_detail::Box;

struct TextLine { std::string value; double x, baseline, width; };
struct PlacedText {
    std::vector<TextLine> lines;
    double size{}, descent{}, font_ascent{};
    Box bounds{};
};

enum class Alignment { left, center, right };
enum class VerticalAlignment { baseline, top, center_baseline };

void translate_text(PlacedText& placed, double dx, double dy) {
    placed.bounds.left += dx; placed.bounds.right += dx;
    placed.bounds.top += dy; placed.bounds.bottom += dy;
    for (auto& line : placed.lines) { line.x += dx; line.baseline += dy; }
}

PlacedText place_text(cairo_t* context, double x, double y, const std::string& value,
                      double size = label_size, Alignment alignment = Alignment::left,
                      VerticalAlignment vertical = VerticalAlignment::baseline) {
    PlacedText result{};
    result.size = size;
    result.bounds = {x, y, x, y};
    if (value.empty()) return result;
    cairo_set_font_size(context, size);
    cairo_text_extents_t font{};
    cairo_text_extents(context, "lp", &font);
    const auto font_descent = font.height + font.y_bearing;
    result.font_ascent = font.height - font_descent;
    // Text._get_layout uses a logical box and the "lp" ascent/descent. Cairo
    // supplies native glyph metrics; its advance retains leading/trailing
    // spaces, unlike ink bounds. Agg/FreeType hinting parity remains unverified.
    double cursor = 0, width = 0, last_baseline = 0;
    std::size_t start = 0;
    for (;;) {
        ProcessInterrupt::check();
        const auto end = value.find('\n', start);
        auto line = value.substr(start, end == std::string::npos ? end : end - start);
        cairo_text_extents_t metrics{};
        if (!line.empty()) cairo_text_extents(context, line.c_str(), &metrics);
        const auto height = std::max(metrics.height, font.height);
        const auto descent = std::max(metrics.height + metrics.y_bearing, font_descent);
        const auto ascent = height - descent;
        last_baseline = ascent + cursor;
        if (result.lines.empty()) cursor = ascent;
        else cursor += std::max(result.font_ascent * 1.2, ascent * 1.2);
        result.lines.push_back({std::move(line), 0, cursor, metrics.x_advance});
        width = std::max(width, metrics.x_advance);
        cursor += descent;
        result.descent = descent;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    const auto left = alignment == Alignment::center ? x - width / 2 :
                      alignment == Alignment::right ? x - width : x;
    auto top = y - (cursor - result.descent);
    if (vertical == VerticalAlignment::top) top = y;
    else if (vertical == VerticalAlignment::center_baseline) top = y - last_baseline / 2;
    result.bounds = {left, top, left + width, top + cursor};
    for (auto& line : result.lines) {
        line.x = left;
        if (alignment == Alignment::center) line.x += (width - line.width) / 2;
        else if (alignment == Alignment::right) line.x += width - line.width;
        line.baseline += top;
    }
    return result;
}

void draw_text(cairo_t* context, const PlacedText& placed) {
    if (placed.lines.empty()) return;
    cairo_set_font_size(context, placed.size);
    for (const auto& line : placed.lines) {
        if (line.value.empty()) continue;
        cairo_move_to(context, line.x, line.baseline);
        cairo_show_text(context, line.value.c_str());
    }
}

struct TickPlacement { double position; PlacedText label; };

struct PanelLayout {
    Box axes, decorations;
    plot_detail::Limits x_view, y_view;
    std::vector<TickPlacement> x_ticks, y_ticks;
    PlacedText x_offset, y_offset, title;
    std::vector<std::size_t> legend;
    std::vector<PlacedText> legend_text;
    std::vector<double> legend_handle_y;
    Box legend_frame{};
};

PanelLayout prepare_panel(cairo_t* context, const RdPanel& panel, Box axes, bool layout_only) {
    double x0 = std::numeric_limits<double>::infinity(), x1 = -x0, y0 = x0, y1 = -x0;
    for (const auto& series : panel.series) {
        ProcessInterrupt::check();
        if (series.rate.size() != series.quality.size()) throw std::invalid_argument("RD curve coordinate counts differ");
        for (std::size_t index = 0; index < series.rate.size(); ++index) {
            if (!std::isfinite(series.rate[index]) || !std::isfinite(series.quality[index])) continue;
            x0 = std::min(x0, series.rate[index]); x1 = std::max(x1, series.rate[index]);
            y0 = std::min(y0, series.quality[index]); y1 = std::max(y1, series.quality[index]);
        }
    }
    PanelLayout result{};
    result.axes = result.decorations = axes;
    result.x_view = panel.series.empty() ? plot_detail::Limits{0, 1} : plot_detail::linear_limits(x0, x1);
    result.y_view = panel.series.empty() ? plot_detail::Limits{0, 1} : plot_detail::linear_limits(y0, y1);
    result.x_view = plot_detail::zero_left(result.x_view.second);
    const auto px = [&](double value) { return axes.left + (value - result.x_view.first) /
        (result.x_view.second - result.x_view.first) * axes.width(); };
    const auto py = [&](double value) { return axes.bottom - (value - result.y_view.first) /
        (result.y_view.second - result.y_view.first) * axes.height(); };
    const auto include_text = [&](const PlacedText& placed) {
        if (placed.bounds.width() > 0 && placed.bounds.height() > 0) result.decorations.include(placed.bounds);
    };
    const auto x_ticks = plot_detail::auto_ticks(result.x_view, axes.width(), true);
    const auto y_ticks = plot_detail::auto_ticks(result.y_view, axes.height(), false);
    const auto x_labels = plot_detail::scalar_labels(x_ticks, result.x_view);
    const auto y_labels = plot_detail::scalar_labels(y_ticks, result.y_view);
    double label_bottom = axes.bottom;
    for (std::size_t index = 0; index < x_ticks.size(); ++index) {
        if (!plot_detail::tick_visible(x_ticks[index], result.x_view)) continue;
        const auto x = px(x_ticks[index]);
        auto placed = place_text(context, x, axes.bottom + 7 * point_unit, x_labels.ticks[index],
                                 label_size, Alignment::center, VerticalAlignment::top);
        label_bottom = std::max(label_bottom, placed.bounds.bottom);
        include_text(placed);
        result.x_ticks.push_back({x, std::move(placed)});
    }
    for (std::size_t index = 0; index < y_ticks.size(); ++index) {
        if (!plot_detail::tick_visible(y_ticks[index], result.y_view)) continue;
        const auto y = py(y_ticks[index]);
        auto placed = place_text(context, axes.left - 7 * point_unit, y,
                                 y_labels.ticks[index], label_size, Alignment::right, VerticalAlignment::center_baseline);
        include_text(placed);
        result.y_ticks.push_back({y, std::move(placed)});
    }
    // Visible bottom/left spines include their outward major marks in get_tightbbox.
    if (!result.x_ticks.empty()) result.decorations.bottom = std::max(result.decorations.bottom, axes.bottom + 3.5 * point_unit);
    if (!result.y_ticks.empty()) result.decorations.left = std::min(result.decorations.left, axes.left - 3.5 * point_unit);
    if (!x_labels.annotation.empty()) {
        result.x_offset = place_text(context, axes.right, label_bottom + 3 * point_unit,
                                    x_labels.annotation, label_size, Alignment::right, VerticalAlignment::top);
        include_text(result.x_offset);
    }
    if (!y_labels.annotation.empty()) {
        result.y_offset = place_text(context, axes.left, axes.top - 3 * point_unit, y_labels.annotation);
        include_text(result.y_offset);
    }
    auto title_anchor = axes.top - 6 * point_unit;
    auto title = place_text(context, (axes.left + axes.right) / 2, title_anchor,
                            panel.title, 12 * point_unit, Alignment::center);
    auto title_top = axes.top;
    if (!panel.title.empty() && !y_labels.annotation.empty() && title.bounds.intersects(result.y_offset.bounds))
        title_top = result.y_offset.bounds.top;
    if (title.bounds.bottom > title_top) {
        title_anchor = title_top - 6 * point_unit;
        title = place_text(context, (axes.left + axes.right) / 2, title_anchor,
                           panel.title, 12 * point_unit, Alignment::center);
        if (title.bounds.bottom > title_top) {
            title_anchor -= title.bounds.bottom - title_top;
            title = place_text(context, (axes.left + axes.right) / 2,
                title_anchor, panel.title, 12 * point_unit, Alignment::center);
        }
    }
    if (title.bounds.width() > 0 && title.bounds.height() > 0) {
        auto bounds = title.bounds;
        if (layout_only) {
            // Axes.get_tightbbox(for_layout_only=True) collapses title width to one pixel.
            bounds.left = (bounds.left + bounds.right) / 2 - 0.5;
            bounds.right = bounds.left + 1;
        }
        result.decorations.include(bounds);
    }
    result.title = std::move(title);
    // UF uses one column, line handles, no title, and lower-right placement.
    // Reduce Legend's TextArea/HPacker/VPacker equations to that configuration.
    constexpr double border_pad = 0.4 * label_size;
    constexpr double label_spacing = 0.5 * label_size;
    constexpr double handle_length = 2 * label_size;
    constexpr double handle_height = 0.7 * label_size;
    constexpr double handle_text_pad = 0.8 * label_size;
    constexpr double border_axes_pad = 0.5 * label_size;
    double legend_width = 0, legend_height = 0;
    std::vector<double> row_ascent, row_descent;
    for (std::size_t index = 0; index < panel.series.size(); ++index) {
        const auto& label = panel.series[index].label;
        if (label.empty() || label.front() == '_') continue;
        result.legend.push_back(index);
        auto placed = place_text(context, 0, 0, label);
        // TextArea centers a multiline label's baseline against the handle.
        if (placed.lines.size() > 1) {
            const auto descent = 0.5 * placed.bounds.height() - 0.5 * placed.font_ascent;
            translate_text(placed, 0, descent - placed.descent);
            placed.descent = descent;
        } else {
            const auto ascent = std::max(placed.font_ascent, placed.bounds.height() - placed.descent);
            placed.bounds.top = -ascent;
            placed.bounds.bottom = placed.descent;
        }
        const auto ascent = std::max(handle_height, -placed.bounds.top);
        const auto descent = std::max(0.0, placed.bounds.bottom);
        row_ascent.push_back(ascent); row_descent.push_back(descent);
        legend_height += ascent + descent + label_spacing;
        legend_width = std::max(legend_width, handle_length + handle_text_pad + placed.bounds.width());
        result.legend_text.push_back(std::move(placed));
    }
    if (!result.legend.empty()) legend_height -= label_spacing;
    // An empty reference Legend still has its padded frame. Do not clamp wide
    // legends inside the axes: their measured extent participates in layout.
    legend_width += 2 * border_pad;
    legend_height += 2 * border_pad;
    const auto legend_right = axes.right - border_axes_pad;
    const auto legend_bottom = axes.bottom - border_axes_pad;
    result.legend_frame = {legend_right - legend_width, legend_bottom - legend_height, legend_right, legend_bottom};
    result.decorations.include(result.legend_frame);
    auto row_top = result.legend_frame.top + border_pad;
    for (std::size_t index = 0; index < result.legend.size(); ++index) {
        auto& placed = result.legend_text[index];
        const auto baseline = row_top + row_ascent[index];
        translate_text(placed, result.legend_frame.left + border_pad + handle_length + handle_text_pad, baseline);
        result.legend_handle_y.push_back(baseline - handle_height / 2);
        result.decorations.include(placed.bounds);
        row_top += row_ascent[index] + row_descent[index] + label_spacing;
    }
    check(cairo_status(context));
    return result;
}

void set_color(cairo_t* context, std::size_t index) {
    const auto& color = colors[index % colors.size()];
    cairo_set_source_rgb(context, color[0] / 255.0, color[1] / 255.0, color[2] / 255.0);
}

void legend_frame_path(cairo_t* context, const Box& frame) {
    // FancyBboxPatch(round,pad=0,rounding_size=0.2) uses quadratic
    // Bezier corners. Convert each quadratic to Cairo's cubic control points.
    const auto l = frame.left, r = frame.right, t = frame.top, b = frame.bottom;
    constexpr double radius = 0.2 * label_size;
    cairo_new_path(context);
    cairo_move_to(context, l + radius, t);
    cairo_line_to(context, r - radius, t);
    cairo_curve_to(context, r - radius / 3, t, r, t + radius / 3, r, t + radius);
    cairo_line_to(context, r, b - radius);
    cairo_curve_to(context, r, b - radius / 3, r - radius / 3, b, r - radius, b);
    cairo_line_to(context, l + radius, b);
    cairo_curve_to(context, l + radius / 3, b, l, b - radius / 3, l, b - radius);
    cairo_line_to(context, l, t + radius);
    cairo_curve_to(context, l, t + radius / 3, l + radius / 3, t, l + radius, t);
    cairo_close_path(context);
}

void scatter_marker_path(cairo_t* context, double x, double y) {
    // MarkerStyle('o') uses Path.unit_circle's eight cubic Bezier segments,
    // scaled to the default 36-point-squared scatter size. Preserve its
    // control constants rather than substituting Cairo's arc approximation.
    static const auto curves = [] {
        constexpr double m = 0.2652031;
        const auto h = std::sqrt(0.5), q = h * m;
        return std::array<std::array<double, 6>, 8>{{
            {m, -1, h-q, -h-q, h, -h},
            {h+q, -h+q, 1, -m, 1, 0},
            {1, m, h+q, h-q, h, h},
            {h-q, h+q, m, 1, 0, 1},
            {-m, 1, -h+q, h+q, -h, h},
            {-h-q, h-q, -1, m, -1, 0},
            {-1, -m, -h-q, -h+q, -h, -h},
            {-h+q, -h-q, -m, -1, 0, -1}
        }};
    }();
    constexpr double radius = 3 * point_unit;
    cairo_new_path(context);
    cairo_move_to(context, x, y + radius);
    for (const auto& curve : curves)
        cairo_curve_to(context, x + radius * curve[0], y - radius * curve[1],
                       x + radius * curve[2], y - radius * curve[3],
                       x + radius * curve[4], y - radius * curve[5]);
    cairo_close_path(context);
}

void draw_panel(cairo_t* context, const RdPanel& panel, const PanelLayout& layout) {
    const auto& axes = layout.axes;
    const auto left = axes.left, right = axes.right, top = axes.top, bottom = axes.bottom;
    const auto px = [&](double value) { return left + (value - layout.x_view.first) /
        (layout.x_view.second - layout.x_view.first) * axes.width(); };
    const auto py = [&](double value) { return bottom - (value - layout.y_view.first) /
        (layout.y_view.second - layout.y_view.first) * axes.height(); };
    const auto clip_axes = [&] {
        cairo_save(context);
        cairo_rectangle(context, left, top, axes.width(), axes.height());
        cairo_clip(context);
    };
    // Axes.draw paints its patch, then stable-sorts artists by zorder:
    // scatter=1, each Axis=1.5, lines=2, spines=2.5, title=3, legend=5.
    // Painting each axes patch also covers earlier panels' overlapping artists.
    cairo_set_source_rgb(context, 1, 1, 1);
    cairo_rectangle(context, left, top, axes.width(), axes.height()); cairo_fill(context);
    clip_axes();
    cairo_set_line_width(context, point_unit);  // scatter's patch linewidth
    cairo_set_line_cap(context, CAIRO_LINE_CAP_BUTT);
    cairo_set_line_join(context, CAIRO_LINE_JOIN_ROUND);
    for (std::size_t method = 0; method < panel.series.size(); ++method) {
        ProcessInterrupt::check();
        const auto& series = panel.series[method];
        set_color(context, method);
        for (std::size_t index = 0; index < series.rate.size(); ++index) {
            if (!std::isfinite(series.rate[index]) || !std::isfinite(series.quality[index])) continue;
            scatter_marker_path(context, px(series.rate[index]), py(series.quality[index]));
            cairo_fill_preserve(context); cairo_stroke(context);
        }
    }
    cairo_restore(context);
    // Axis.draw keeps each tick's grid, mark and label together; the offset
    // text follows all ticks. Only the gridline is clipped to the axes patch.
    for (const auto& tick : layout.x_ticks) {
        clip_axes();
        cairo_set_source_rgb(context, 0.5, 0.5, 0.5); cairo_set_line_width(context, 0.5 * point_unit);
        cairo_set_line_cap(context, CAIRO_LINE_CAP_SQUARE);
        cairo_set_line_join(context, CAIRO_LINE_JOIN_ROUND);
        cairo_move_to(context, tick.position, top); cairo_line_to(context, tick.position, bottom); cairo_stroke(context);
        cairo_restore(context);
        cairo_set_source_rgb(context, 0, 0, 0); cairo_set_line_width(context, 0.8 * point_unit);
        cairo_set_line_cap(context, CAIRO_LINE_CAP_BUTT);
        cairo_move_to(context, tick.position, bottom); cairo_line_to(context, tick.position, bottom + 3.5 * point_unit); cairo_stroke(context);
        draw_text(context, tick.label);
    }
    cairo_set_source_rgb(context, 0, 0, 0); draw_text(context, layout.x_offset);
    for (const auto& tick : layout.y_ticks) {
        clip_axes();
        cairo_set_source_rgb(context, 0.5, 0.5, 0.5); cairo_set_line_width(context, 0.5 * point_unit);
        cairo_set_line_cap(context, CAIRO_LINE_CAP_SQUARE);
        cairo_set_line_join(context, CAIRO_LINE_JOIN_ROUND);
        cairo_move_to(context, left, tick.position); cairo_line_to(context, right, tick.position); cairo_stroke(context);
        cairo_restore(context);
        cairo_set_source_rgb(context, 0, 0, 0); cairo_set_line_width(context, 0.8 * point_unit);
        cairo_set_line_cap(context, CAIRO_LINE_CAP_BUTT);
        cairo_move_to(context, left, tick.position); cairo_line_to(context, left - 3.5 * point_unit, tick.position); cairo_stroke(context);
        draw_text(context, tick.label);
    }
    cairo_set_source_rgb(context, 0, 0, 0); draw_text(context, layout.y_offset);
    clip_axes();
    cairo_set_line_cap(context, CAIRO_LINE_CAP_SQUARE);
    cairo_set_line_join(context, CAIRO_LINE_JOIN_ROUND);
    for (std::size_t method = 0; method < panel.series.size(); ++method) {
        ProcessInterrupt::check();
        const auto& series = panel.series[method];
        set_color(context, method); cairo_set_line_width(context, line_width);
        bool active = false;
        for (std::size_t index = 0; index < series.rate.size(); ++index) {
            if (!std::isfinite(series.rate[index]) || !std::isfinite(series.quality[index])) {
                if (active) cairo_stroke(context);
                active = false; continue;
            }
            if (active) cairo_line_to(context, px(series.rate[index]), py(series.quality[index]));
            else { cairo_move_to(context, px(series.rate[index]), py(series.quality[index])); active = true; }
        }
        if (active) cairo_stroke(context);
    }
    cairo_restore(context);
    // Separate projecting-capped spine segments retain the reference order.
    cairo_set_source_rgb(context, 0, 0, 0); cairo_set_line_width(context, 0.8 * point_unit);
    cairo_set_line_cap(context, CAIRO_LINE_CAP_SQUARE);
    cairo_set_line_join(context, CAIRO_LINE_JOIN_MITER);
    cairo_move_to(context, left, bottom); cairo_line_to(context, left, top); cairo_stroke(context);
    cairo_move_to(context, right, bottom); cairo_line_to(context, right, top); cairo_stroke(context);
    cairo_move_to(context, left, bottom); cairo_line_to(context, right, bottom); cairo_stroke(context);
    cairo_move_to(context, left, top); cairo_line_to(context, right, top); cairo_stroke(context);
    draw_text(context, layout.title);
    {
        const auto& frame = layout.legend_frame;
        legend_frame_path(context, frame);
        cairo_set_line_cap(context, CAIRO_LINE_CAP_BUTT);
        cairo_set_line_join(context, CAIRO_LINE_JOIN_MITER);
        cairo_set_source_rgba(context, 1, 1, 1, 0.8);
        cairo_fill_preserve(context);
        cairo_set_source_rgba(context, 0.8, 0.8, 0.8, 0.8);
        cairo_set_line_width(context, point_unit);
        cairo_stroke(context);
        for (std::size_t index = 0; index < layout.legend.size(); ++index) {
            const auto y = layout.legend_handle_y[index];
            set_color(context, layout.legend[index]); cairo_set_line_width(context, line_width);
            cairo_set_line_cap(context, CAIRO_LINE_CAP_SQUARE);
            const auto left = frame.left + 0.4 * label_size;
            cairo_move_to(context, left, y); cairo_line_to(context, left + 2 * label_size, y); cairo_stroke(context);
            cairo_set_source_rgb(context, 0, 0, 0); draw_text(context, layout.legend_text[index]);
        }
    }
    check(cairo_status(context));
}

void layout_warning(plot_detail::LayoutFailure failure) {
    switch (failure) {
    case plot_detail::LayoutFailure::none: return;
    case plot_detail::LayoutFailure::horizontal_margins:
        std::cerr << "warning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.\n";
        return;
    case plot_detail::LayoutFailure::vertical_margins:
        std::cerr << "warning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.\n";
        return;
    case plot_detail::LayoutFailure::axes_width:
        std::cerr << "warning: Tight layout not applied. tight_layout cannot make Axes width small enough to accommodate all Axes decorations\n";
        return;
    }
}

void legend_diagnostic(const RdPanel& panel, bool combined) {
    const auto labeled = std::any_of(panel.series.begin(), panel.series.end(), [](const auto& series) {
        return !series.label.empty() && series.label.front() != '_';
    });
    if (labeled) return;
    // The maintained script has distinct separate/combined legend call sites.
    // Default Python warnings repeat neither within a call site nor across
    // metrics at that site. Custom Python warning filters are not implemented.
    static std::atomic<bool> combined_seen{false}, separate_seen{false};
    auto& seen = combined ? combined_seen : separate_seen;
    if (seen.exchange(true, std::memory_order_relaxed)) return;
    std::cerr << "warning: No artists with labels found to put in legend.  Note that "
                 "artists whose label start with an underscore are ignored "
                 "when legend() is called with no argument.\n";
}

cairo_status_t append_png(void* closure, const unsigned char* data, unsigned int length) noexcept {
    try {
        auto& bytes = *static_cast<std::vector<std::uint8_t>*>(closure);
        bytes.insert(bytes.end(), data, data + length);
        return CAIRO_STATUS_SUCCESS;
    } catch (...) { return CAIRO_STATUS_WRITE_ERROR; }
}

std::uint32_t crc32(const std::uint8_t* bytes, std::size_t size) {
    std::uint32_t crc = 0xffffffffu;
    for (std::size_t index = 0; index < size; ++index) {
        crc ^= bytes[index];
        for (unsigned bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return crc ^ 0xffffffffu;
}
void big_endian(std::uint8_t* bytes, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) bytes[index] = static_cast<std::uint8_t>(value >> (24 - index * 8));
}
void write_png(const std::vector<std::uint8_t>& bytes, const std::filesystem::path& output) {
    if (bytes.size() < 33) throw std::runtime_error("Cairo returned an incomplete PNG");
    std::array<std::uint8_t, 21> density{};
    big_endian(density.data(), 9);
    density[4] = 'p'; density[5] = 'H'; density[6] = 'Y'; density[7] = 's';
    big_endian(density.data() + 8, 11811); big_endian(density.data() + 12, 11811);
    density[16] = 1; big_endian(density.data() + 17, crc32(density.data() + 4, 13));
    std::ofstream file(output, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Cannot create RD curve PNG: " + output.string());
    file.write(reinterpret_cast<const char*>(bytes.data()), 8);
    for (std::size_t position = 8; position < bytes.size();) {
        ProcessInterrupt::check();
        if (bytes.size() - position < 12) throw std::runtime_error("Truncated Cairo PNG chunk");
        std::uint32_t length = 0;
        for (unsigned index = 0; index < 4; ++index) length = (length << 8) | bytes[position + index];
        if (length > bytes.size() - position - 12) throw std::runtime_error("Invalid Cairo PNG chunk size");
        const std::string type(reinterpret_cast<const char*>(bytes.data() + position + 4), 4);
        if (type != "pHYs") file.write(reinterpret_cast<const char*>(bytes.data() + position), static_cast<std::size_t>(length) + 12);
        if (type == "IHDR") file.write(reinterpret_cast<const char*>(density.data()), density.size());
        position += static_cast<std::size_t>(length) + 12;
    }
    file.close();
    if (!file) throw std::runtime_error("Cannot write RD curve PNG");
}

}  // namespace
#endif

void write_rd_plot(const std::vector<RdPanel>& panels, const std::filesystem::path& output, bool combined) {
#if defined(UF_HAS_CAIRO) && CAIRO_HAS_PNG_FUNCTIONS
    if (panels.empty() || (!combined && panels.size() != 1)) throw std::invalid_argument("Invalid RD plot panel count");
    const auto panel_width = combined ? 500u : 640u, panel_height = combined ? 400u : 480u;
    if (panels.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) / (panel_width * 3)) {
        throw std::overflow_error("RD plot width overflow");
    }
    Surface surface(cairo_image_surface_create(CAIRO_FORMAT_ARGB32, static_cast<int>(panels.size() * panel_width * 3),
                                              panel_height * 3), cairo_surface_destroy);
    check(cairo_surface_status(surface.get()));
    Context context(cairo_create(surface.get()), cairo_destroy); check(cairo_status(context.get()));
    cairo_select_font_face(context.get(), "DejaVu Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    const auto figure_width = static_cast<double>(panels.size() * panel_width);
    auto axes = plot_detail::initial_row(figure_width, panel_height, panels.size());
    for (const auto& panel : panels) { ProcessInterrupt::check(); legend_diagnostic(panel, combined); }
    if (combined) {
        // fig.tight_layout measures at the initial figure dpi (100), before
        // savefig switches to 300 dpi. This pass only queries Cairo text metrics.
        std::vector<Box> decorations;
        decorations.reserve(panels.size());
        for (std::size_t index = 0; index < panels.size(); ++index) {
            ProcessInterrupt::check();
            decorations.push_back(prepare_panel(context.get(), panels[index], axes[index], true).decorations);
        }
        auto adjusted = plot_detail::tight_row(figure_width, panel_height, axes, decorations);
        layout_warning(adjusted.failure);
        axes = std::move(adjusted.axes);
    }
    cairo_scale(context.get(), 3, 3);
    cairo_set_source_rgb(context.get(), 1, 1, 1); cairo_paint(context.get());
    for (std::size_t index = 0; index < panels.size(); ++index) {
        ProcessInterrupt::check();
        cairo_save(context.get());
        const auto layout = prepare_panel(context.get(), panels[index], axes[index], false);
        draw_panel(context.get(), panels[index], layout);
        cairo_restore(context.get());
    }
    std::vector<std::uint8_t> bytes;
    check(cairo_surface_write_to_png_stream(surface.get(), append_png, &bytes));
    write_png(bytes, output);
#else
    (void)panels; (void)output; (void)combined;
    throw std::runtime_error("RD PNG plotting requires Cairo development files with PNG support");
#endif
}

}  // namespace uf::evaluation
