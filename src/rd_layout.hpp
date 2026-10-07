// One-row UF subplot layout adapted from Matplotlib 3.10.9 GridSpec/tight_layout.
// Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved.
// Retained license/modifications: licenses/Matplotlib.txt, licenses/README.md.
#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uf::evaluation::plot_detail {

// Drawing pixels at 100 dpi, with the origin at the upper left.
struct Box {
    double left, top, right, bottom;
    double width() const { return right - left; }
    double height() const { return bottom - top; }
    void include(const Box& other) {
        left = std::min(left, other.left); top = std::min(top, other.top);
        right = std::max(right, other.right); bottom = std::max(bottom, other.bottom);
    }
    bool intersects(const Box& other) const {
        return left <= other.right && right >= other.left && top <= other.bottom && bottom >= other.top;
    }
};

inline std::vector<Box> initial_row(double width, double height, std::size_t columns) {
    if (!columns) throw std::invalid_argument("RD subplot row must contain an axes");
    const auto count = static_cast<double>(columns);
    const auto cell = (0.9 - 0.125) / (count + 0.2 * (count - 1));
    const auto gap = 0.2 * cell;
    const auto normalized = cell * count / count;
    std::vector<Box> result;
    result.reserve(columns);
    double accumulated = 0;
    for (std::size_t index = 0; index < columns; ++index) {
        if (index) accumulated += gap;
        const auto left = (0.125 + accumulated) * width;
        accumulated += normalized;
        result.push_back({left, (1 - 0.88) * height, (0.125 + accumulated) * width, (1 - 0.11) * height});
    }
    return result;
}

enum class LayoutFailure { none, horizontal_margins, vertical_margins, axes_width };

struct RowLayout {
    std::vector<Box> axes;
    LayoutFailure failure = LayoutFailure::none;
};

inline RowLayout tight_row(double width, double height, const std::vector<Box>& axes,
                           const std::vector<Box>& decorations) {
    if (axes.empty() || axes.size() != decorations.size())
        throw std::invalid_argument("RD subplot bounds and decorations differ");
    // Default pad=1.08 times font.size=10 points, converted to inches/figure size.
    constexpr double pad_inches = 1.08 * (10.0 / 72.0);
    const auto pad_x = pad_inches / (width / 100.0), pad_y = pad_inches / (height / 100.0);
    const auto margin_left = std::max((axes.front().left - decorations.front().left) / width, 0.0) + pad_x;
    const auto initial_margin_right = std::max((decorations.back().right - axes.back().right) / width, 0.0) + pad_x;
    double top_space = 0, bottom_space = 0;
    double inner_space = 0;
    for (std::size_t index = 0; index < axes.size(); ++index) {
        top_space = std::max(top_space, (axes[index].top - decorations[index].top) / height);
        bottom_space = std::max(bottom_space, (decorations[index].bottom - axes[index].bottom) / height);
        if (index) inner_space = std::max(inner_space,
            (decorations[index - 1].right - axes[index - 1].right) / width +
            (axes[index].left - decorations[index].left) / width);
    }
    const auto initial_margin_top = std::max(top_space, 0.0) + pad_y;
    const auto margin_bottom = std::max(bottom_space, 0.0) + pad_y;
    // A failed adjustment retains GridSpec's initial positions, as tight_layout does.
    if (margin_left + initial_margin_right >= 1) return {axes, LayoutFailure::horizontal_margins};
    if (margin_bottom + initial_margin_top >= 1) return {axes, LayoutFailure::vertical_margins};
    const auto count = static_cast<double>(axes.size());
    const auto gap = axes.size() > 1 ? inner_space + pad_x : 0.0;
    const auto first_cell = (1 - initial_margin_right - margin_left - gap * (count - 1)) / count;
    if (first_cell < 0) return {axes, LayoutFailure::axes_width};
    // TightLayoutEngine passes rect=(0,0,1,1). get_tight_layout_figure
    // repeats its calculation against the same initial axes/decorations with
    // adjusted rect edges, before changing any axes positions. Retain the
    // right/top complement boundaries rather than simplifying 1-(1-x).
    const auto first_right = 1 - initial_margin_right, first_top = 1 - initial_margin_top;
    const auto rect_right = 1 - (1 - first_right), rect_top = 1 - (1 - first_top);
    const auto margin_right = 1 - rect_right, margin_top = 1 - rect_top;
    if (margin_left + margin_right >= 1) return {axes, LayoutFailure::horizontal_margins};
    if (margin_bottom + margin_top >= 1) return {axes, LayoutFailure::vertical_margins};
    const auto cell = (1 - margin_right - margin_left - gap * (count - 1)) / count;
    if (cell < 0) return {axes, LayoutFailure::axes_width};
    // GridSpec rebuilds a grid from wspace=gap/cell after subplots_adjust.
    const auto relative_gap = axes.size() > 1 ? gap / cell : 0.0;
    const auto rebuilt_cell = (1 - margin_right - margin_left) / (count + relative_gap * (count - 1));
    const auto rebuilt_gap = relative_gap * rebuilt_cell;
    const auto normalized = rebuilt_cell * count / count;
    std::vector<Box> adjusted;
    adjusted.reserve(axes.size());
    double accumulated = 0;
    for (std::size_t index = 0; index < axes.size(); ++index) {
        if (index) accumulated += rebuilt_gap;
        const auto left = (margin_left + accumulated) * width;
        accumulated += normalized;
        adjusted.push_back({left, margin_top * height, (margin_left + accumulated) * width, (1 - margin_bottom) * height});
    }
    return {std::move(adjusted), LayoutFailure::none};
}

}  // namespace uf::evaluation::plot_detail
