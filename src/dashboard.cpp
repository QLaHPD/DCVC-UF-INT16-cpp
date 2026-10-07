#include "uf/dashboard.hpp"
#include "uf/events.hpp"
#include "uf/process.hpp"
#include "uf/python_json.hpp"
#include "uf/python_path.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cwchar>
#include <deque>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <poll.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

namespace uf {
namespace {
using Clock = std::chrono::steady_clock;
using WallClock = std::chrono::system_clock;
using Json = nlohmann::json;

std::string text_field(const Json& row, const char* field) {
    const auto found = row.find(field);
    if (found == row.end() || found->is_null()) return {};
    return found->is_string() ? found->get<std::string>() : python_json_repr(*found);
}

std::string duration_text(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0) return "--:--:--";
    const auto value = static_cast<std::uint64_t>(seconds);
    std::ostringstream out;
    out << std::setfill('0') << std::setw(2) << value / 3600 << ':'
        << std::setw(2) << (value / 60) % 60 << ':' << std::setw(2) << value % 60;
    return out.str();
}

std::string bytes_text(std::uint64_t bytes) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double amount = static_cast<double>(bytes);
    int unit = 0;
    while (amount >= 1024 && unit < 4) { amount /= 1024; ++unit; }
    std::ostringstream out;
    if (unit == 0) out << bytes;
    else out << std::fixed << std::setprecision(1) << amount;
    out << units[unit];
    return out.str();
}

std::string count_text(std::uint64_t count) {
    auto value = std::to_string(count);
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(value.size()) - 3; i > 0; i -= 3)
        value.insert(static_cast<std::size_t>(i), ",");
    return value;
}

// Escape controls and invalid UTF-8; clip by display cells so event text cannot
// move the cursor or break the terminal layout. wcwidth follows LC_CTYPE.
std::string clip(const std::string& text, int width) {
    if (width <= 0) return {};
    std::vector<std::pair<std::string, int>> cells;
    std::mbstate_t state{};
    int total = 0;
    for (std::size_t offset = 0; offset < text.size();) {
        wchar_t value{};
        auto length = std::mbrtowc(&value, text.data() + offset, text.size() - offset, &state);
        std::string part;
        int size = 1;
        if (length == static_cast<std::size_t>(-1) || length == static_cast<std::size_t>(-2)) {
            length = 1; part = "?"; state = {};
        } else if (length == 0) {
            length = 1; part = "?";
        } else if (value < 32 || (value >= 127 && value < 160)) {
            part = " ";
        } else {
            size = ::wcwidth(value);
            if (size < 0) size = 1;
            part = text.substr(offset, length);
        }
        offset += length;
        total += size;
        cells.emplace_back(std::move(part), size);
    }
    const bool shortened = total > width;
    const int limit = shortened && width > 3 ? width - 3 : width;
    std::string result;
    int used = 0;
    for (const auto& cell : cells) {
        if (used + cell.second > limit) break;
        result += cell.first;
        used += cell.second;
    }
    if (shortened && width > 3) result += "...";
    return result;
}

bool output_text(const std::string& value) noexcept {
    std::size_t offset = 0;
    while (offset < value.size()) {
        const auto amount = ::write(STDOUT_FILENO, value.data() + offset, value.size() - offset);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) return false;
        offset += static_cast<std::size_t>(amount);
    }
    return true;
}

bool approval_status(const std::string& status) {
    return status == "awaiting-approval" || status == "validation-failed" ||
           status == "cleanup-failed" || status == "validate-failed";
}

}  // namespace

struct Dashboard::Impl {
    struct Job {
        int slot{};
        Clock::time_point started{Clock::now()};
        Json record = Json::object();
        std::string text;
    };
    std::filesystem::path storage;
    int worker_count;
    // EncodeDashboard uses time.time() for overall elapsed time; per-job
    // completion samples and input timeouts retain time.monotonic().
    WallClock::time_point started{WallClock::now()};
    mutable std::mutex mutex;
    std::atomic<bool> stop{false}, tui{false};
    std::thread thread;
    termios saved{};
    bool terminal_saved{};
    std::uint64_t total{}, done{}, samples{};
    double sample_seconds{};
    std::map<std::string, Job> jobs;
    std::vector<Json> channels;
    std::deque<std::string> messages;
    std::map<std::string, std::string> plain_status;
    std::vector<DashboardAction> actions;
    int view{}, selected{};
    std::optional<std::string> confirming;
    std::string input;
    Clock::time_point escape_started{};

    Impl(std::filesystem::path path, int workers, const std::string& mode)
        : storage(std::move(path)), worker_count(std::max(1, workers)) {
        if (mode != "auto" && mode != "plain" && mode != "tui")
            throw std::invalid_argument("--ui must be auto, plain, or tui");
        const char* term = std::getenv("TERM");
        const bool available = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO) &&
                               (!term || std::string(term) != "dumb");
        if (mode == "plain" || (mode == "auto" && !available)) return;
        if (!available || tcgetattr(STDIN_FILENO, &saved) != 0) {
            if (mode == "tui") throw std::runtime_error("Cannot initialize the terminal dashboard");
            return;
        }
        auto raw = saved;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
            if (mode == "tui") throw std::runtime_error("Cannot set dashboard terminal mode");
            return;
        }
        terminal_saved = true;
        if (!output_text("\033[?1049h\033[?25l\033[H")) {
            restore_terminal();
            if (mode == "tui") throw std::runtime_error("Cannot draw the terminal dashboard");
            return;
        }
        tui.store(true);
    }

    ~Impl() { close(); }
    void restore_terminal() noexcept {
        if (terminal_saved) {
            output_text("\033[?25h\033[?1049l");
            tcsetattr(STDIN_FILENO, TCSANOW, &saved);
            terminal_saved = false;
        }
        tui.store(false);
    }
    void close() noexcept {
        stop.store(true);
        if (thread.joinable()) thread.join();
        restore_terminal();
    }

    void start() {
        if (!tui.load()) return;
        try { thread = std::thread([this] { loop(); }); }
        catch (...) { restore_terminal(); throw; }
    }

    std::vector<const Json*> candidates() const {
        std::vector<const Json*> result;
        for (const auto& row : channels)
            if (approval_status(row.at("status").get<std::string>())) result.push_back(&row);
        return result;
    }

    void update(const Json& record) {
        std::lock_guard<std::mutex> lock(mutex);
        if (stop.load()) return;
        const auto kind = text_field(record, "event");
        auto key = text_field(record, "job_key");
        if (key.empty()) key = text_field(record, "source");
        if (kind == "run_discovered") total = record.value("total", std::uint64_t{});
        if (kind == "lifecycle_inventory") {
            channels.clear();
            for (const auto& row : record.at("channels")) {
                const auto count = row.at("archives").size();
                const std::string status = row.value("blocked", false) ? "validation-failed"
                    : row.value("originals", std::uint64_t{}) ? "awaiting-approval"
                    : row.value("ineligible", std::uint64_t{}) ? "retained"
                    : row.value("absent", std::uint64_t{}) ? "cleaned" : "complete";
                const auto channel = text_field(row, "channel");
                channels.push_back({{"channel_id", channel}, {"status", status},
                    {"processed", count}, {"pending", 0}, {"already_done", 0},
                    {"failures", row.value("blocked", false) ? 1 : 0},
                    {"total", count}, {"delete_files", row.value("originals", std::uint64_t{})},
                    {"reclaimable_bytes", row.value("bytes", std::uint64_t{})},
                    {"last_error", text_field(row, "last_error")}});
                if (!tui.load() && plain_status[channel] != status &&
                    (approval_status(status) || status == "cleaned"))
                    output_text(clip(channel + ": " + status, 4096) + "\n");
                plain_status[channel] = status;
            }
            selected = std::min(selected, std::max(0, static_cast<int>(candidates().size()) - 1));
        }
        if (!key.empty() && (kind == "encode_started" || kind == "encode_progress" ||
                             kind == "validation_started")) {
            auto found = jobs.find(key);
            if (found == jobs.end()) {
                int slot = 0;
                for (;;) {
                    bool used = false;
                    for (const auto& entry : jobs) if (entry.second.slot == slot) used = true;
                    if (!used) break;
                    ++slot;
                }
                found = jobs.emplace(key, Job{}).first;
                found->second.slot = slot;
            }
            auto& job = found->second;
            job.record.update(record);
            auto source = text_field(job.record, "source");
            if (source.empty()) source = key;
            std::ostringstream text;
            text << python_path_name_text(source) << ": "
                 << (kind == "validation_started" ? "validating" : "encode") << ' '
                 << count_text(job.record.value("frames", std::uint64_t{})) << " frames "
                 << std::fixed << std::setprecision(2) << job.record.value("fps", 0.0) << " FPS";
            job.text = text.str();
            if (!tui.load() && kind == "encode_progress")
                output_text(clip(job.text, 4096) + "\n");
        }
        if (kind == "encode_completed" || kind == "already_encoded" || kind == "encode_failed") {
            ++done;
            const auto found = jobs.find(key);
            if (found != jobs.end()) {
                const auto elapsed = std::chrono::duration<double>(Clock::now() - found->second.started).count();
                if (std::isfinite(elapsed) && elapsed > 0) { sample_seconds += elapsed; ++samples; }
                jobs.erase(found);
            }
        }
        if (kind != "encode_progress" && kind != "lifecycle_inventory") {
            auto stamp = text_field(record, "time");
            stamp = stamp.size() >= 19 ? stamp.substr(11, 8) : "";
            std::string detail;
            for (const auto* field : {"error", "message", "resolution", "source", "archive"}) {
                detail = text_field(record, field);
                if (!detail.empty()) break;
            }
            const auto message = stamp + " " + kind + ": " + detail;
            messages.push_back(message);
            if (messages.size() > 40) messages.pop_front();
            if (!tui.load()) output_text(clip(message, 4096) + "\n");
        }
    }

    void key(int value) {
        if (confirming) {
            if (value == 'y' || value == 'Y') {
                actions.push_back({"approve", confirming}); confirming.reset();
            } else if (value == 'n' || value == 'N' || value == 27) confirming.reset();
            return;
        }
        if (value == 9 || value == 1002) view = (view + 1) % 4;
        else if (value == 1003) view = (view + 3) % 4;
        else if (value == 'w' || value == 'W') view = 0;
        else if (value == 'c' || value == 'C') view = 1;
        else if (value == 'a' || value == 'A') view = 2;
        else if (value == 'e' || value == 'E') view = 3;
        else if (value == 1000) selected = std::max(0, selected - 1);
        else if (value == 1001) ++selected;
        else if (value == 'x' || value == 'X') actions.push_back({"exit-approvals", std::nullopt});
        else {
            const auto rows = candidates();
            if (rows.empty()) return;
            selected = std::min(selected, static_cast<int>(rows.size()) - 1);
            const auto& row = *rows[selected];
            const auto channel = row.at("channel_id").get<std::string>();
            const bool awaiting = row.at("status") == "awaiting-approval";
            if ((value == 'd' || value == 'D') && awaiting) confirming = channel;
            else if ((value == 'k' || value == 'K') && awaiting) actions.push_back({"keep", channel});
            else if (value == 'r' || value == 'R') actions.push_back({"retry", channel});
        }
        // poll_actions lets Down move beyond the current rows. Clamp only on
        // lifecycle refresh or when a selected-row action resolves its target.
    }

    void parse_input() {
        while (!input.empty()) {
            if (input[0] != '\033') {
                key(static_cast<unsigned char>(input[0])); input.erase(0, 1); continue;
            }
            if (input.size() == 1) {
                if (Clock::now() - escape_started < std::chrono::milliseconds(35)) return;
                key(27); input.erase(0, 1); continue;
            }
            if (input[1] != '[' && input[1] != 'O') {
                key(27); input.erase(0, 1); continue;
            }
            std::size_t end = 2;
            while (end < input.size() && (input[end] < '@' || input[end] > '~')) ++end;
            if (end == input.size()) {
                if (Clock::now() - escape_started < std::chrono::milliseconds(100)) return;
                input.clear(); key(27); return;
            }
            if (input[end] == 'A') key(1000);
            if (input[end] == 'B') key(1001);
            if (input[end] == 'C') key(1002);
            if (input[end] == 'D') key(1003);
            input.erase(0, end + 1);
        }
    }

    bool draw() {
        winsize size{};
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) != 0 || !size.ws_row || !size.ws_col) return true;
        const int width = std::max(0, static_cast<int>(size.ws_col) - 1);
        const int height = size.ws_row;
        const auto elapsed = std::chrono::duration<double>(WallClock::now() - started).count();
        const double average = samples ? sample_seconds / samples : -1;
        const double eta = samples ? average * (total > done ? total - done : 0) / worker_count : -1;
        int approvals = 0, failures = 0;
        std::uint64_t active = 0;
        for (const auto& row : channels) {
            if (row.at("status") == "awaiting-approval") ++approvals;
            failures += row.value("failures", 0);
            active += row.value("active", std::uint64_t{});
        }
        std::error_code error;
        const auto space = std::filesystem::space(storage, error);
        const auto free = error ? std::string{} : " free=" + bytes_text(space.available);
        std::ostringstream header;
        header << "DCVC managed encode  " << std::fixed << std::setprecision(1)
               << std::setw(5) << (total ? 100.0 * done / total : 100.0) << "% " << done << '/' << total
               << "  elapsed=" << duration_text(std::max(0.0, elapsed)) << " avg/video=" << duration_text(average)
               << " eta=" << duration_text(eta);
        static const char* views[] = {"workers", "channels", "approvals", "events"};
        std::vector<std::string> lines = {header.str(),
            "active=" + std::to_string(active) + " failures=" + std::to_string(failures) +
            " approvals=" + std::to_string(approvals) + free,
            "Views: [w]orkers [c]hannels [a]pprovals [e]vents  Tab/arrow switch",
            std::string("View: ") + views[view]};
        if (confirming) lines.push_back("DELETE " + *confirming + "? Press y to confirm, n/Esc to cancel.");
        const int body_height = std::max(0, height - static_cast<int>(lines.size()) - 1);
        std::vector<std::string> body;
        if (view == 0) {
            std::map<int, std::string> workers;
            for (int i = 0; i < worker_count; ++i) workers[i] = "[" + std::to_string(i) + "] idle";
            for (const auto& entry : jobs) workers[entry.second.slot] = entry.second.text;
            for (const auto& entry : workers)
                body.push_back("worker " + std::to_string(entry.first) + ": " + entry.second);
        } else if (view == 1) {
            for (const auto& row : channels)
                body.push_back(text_field(row, "channel_id") + "  " + text_field(row, "status") +
                    "  processed=" + text_field(row, "processed") + "/0 preexisting=0 failures=" + text_field(row, "failures"));
            if (body.empty()) body.push_back("No channels.");
        } else if (view == 2) {
            const auto rows = candidates();
            for (std::size_t index = 0; index < rows.size(); ++index) {
                const auto& row = *rows[index];
                body.push_back(std::string(index == static_cast<std::size_t>(selected) ? "> " : "  ") +
                    text_field(row, "channel_id") + "  " + text_field(row, "status") + "  files=" +
                    text_field(row, "delete_files") + " reclaim=" +
                    bytes_text(row.value("reclaimable_bytes", std::uint64_t{})) + " " + text_field(row, "last_error"));
            }
            if (body.empty()) body.push_back("No cleanup approvals or cleanup failures.");
            body.push_back("Keys: Up/Down select, d delete, k keep, r retry validation, x leave pending and exit after encoding.");
        } else {
            const auto first = messages.size() > static_cast<std::size_t>(body_height)
                ? messages.size() - body_height : 0;
            for (std::size_t index = first; index < messages.size(); ++index) body.push_back(messages[index]);
            if (body.empty()) body.push_back("No lifecycle events yet.");
        }
        for (int i = 0; i < body_height && i < static_cast<int>(body.size()); ++i) lines.push_back(body[i]);
        lines.resize(static_cast<std::size_t>(height - 1));
        lines.push_back("Ctrl-C safely stops encoding; in-progress outputs are discarded atomically.");
        std::string screen = "\033[H";
        for (int row = 0; row < height; ++row) {
            if (row) screen += "\r\n";
            screen += "\033[2K" + clip(lines[row], width);
        }
        return output_text(screen);
    }

    void loop() noexcept {
        try {
            while (!stop.load() && !ProcessInterrupt::requested()) {
                pollfd descriptor{STDIN_FILENO, POLLIN, 0};
                const auto ready = ::poll(&descriptor, 1, 100);
                if (ready < 0 && errno != EINTR) break;
                if (ready > 0 && (descriptor.revents & (POLLHUP | POLLERR | POLLNVAL))) break;
                std::lock_guard<std::mutex> lock(mutex);
                if (ready > 0 && (descriptor.revents & POLLIN)) {
                    char buffer[256];
                    const auto count = ::read(STDIN_FILENO, buffer, sizeof(buffer));
                    if (count > 0) {
                        if (input.empty()) escape_started = Clock::now();
                        input.append(buffer, static_cast<std::size_t>(count));
                    } else if (count < 0 && errno != EINTR && errno != EAGAIN) break;
                }
                parse_input();
                if (!draw()) break;
            }
        } catch (...) {
            // A presentation failure must not leave a raw terminal behind.
        }
        std::lock_guard<std::mutex> lock(mutex);
        restore_terminal();
    }
};

Dashboard::Dashboard(std::filesystem::path storage, int workers, std::string mode) {
    // Initialize before workflow threads start; only character classification changes.
    std::setlocale(LC_CTYPE, "");
    impl_ = std::make_shared<Impl>(std::move(storage), workers, mode);
    impl_->start();
    const std::weak_ptr<Impl> weak = impl_;
    set_event_sink([weak](const Json& record) {
        if (const auto state = weak.lock()) state->update(record);
    });
}
Dashboard::~Dashboard() {
    set_event_sink({});
    impl_->close();
}
bool Dashboard::interactive() const { return impl_->tui.load(); }
std::vector<DashboardAction> Dashboard::poll_actions() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<DashboardAction> result;
    result.swap(impl_->actions);
    return result;
}

}  // namespace uf
