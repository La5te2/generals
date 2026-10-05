// external strategy/agents communication: start a strategy process, exchange observations and actions through pipes, and close it after the game.
// a worker thread handles blocking reads and writes, keeping game updates and drawing responsive while waiting for a reply.
#include "process.hpp"
#include <array>
#include <condition_variable>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <pthread.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace {
    // parse arguments directly rather than passing user text through a command shell.
    std::vector<std::string> arguments(std::string_view command) {
        std::vector<std::string> result;
        std::string word;
        char quote = 0;
        bool started = false;
        for (char letter : command) {
            if (quote) {
                if (letter == quote) quote = 0;
                else word += letter;
            } else if (letter == '"' || letter == '\'') {
                quote = letter;
                started = true;
            } else if (letter == ' ' || letter == '\t') {
                if (started) { result.push_back(word); word.clear(); started = false; }
            } else { word += letter; started = true; }
        }
        if (quote) return {};
        if (started) result.push_back(word);
        if (!result.empty() && result.front().empty()) return {};
        return result;
    }

#ifdef _WIN32
    void close(HANDLE& handle) {
        if (handle) CloseHandle(handle);
        handle = nullptr;
    }

    std::wstring wide(std::string_view text) {
        int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), nullptr, 0);
        if (!length) return {};
        std::wstring result(length, L'\0');
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), result.data(), length);
        return result;
    }

    // Windows receives one command line. escape quotes and trailing backslashes using CRT argument rules.
    std::wstring quoted(const std::wstring& argument) {
        std::wstring result = L"\"";
        std::size_t slashes = 0;
        for (wchar_t letter : argument) {
            if (letter == L'\\') { ++slashes; continue; }
            result.append(letter == L'"' ? slashes * 2 + 1 : slashes, L'\\');
            result += letter;
            slashes = 0;
        }
        result.append(slashes * 2, L'\\');
        return result + L'"';
    }
#else
    void close(int& descriptor) {
        if (descriptor >= 0) ::close(descriptor);
        descriptor = -1;
    }
#endif
}

namespace NEBULA {
    struct StrategyProcess::Process {
        mutable std::mutex mutex;
        std::condition_variable changed;
        std::thread worker;
        std::optional<Observation> pending;
        std::optional<Action> reply;
        std::uint64_t repliedTick = 0;
        Time received{};
        std::string failure;
        bool closing = false, done = true;
#ifdef _WIN32
        HANDLE child = nullptr, job = nullptr, input = nullptr, output = nullptr;
#else
        pid_t child = -1;
        int input = -1, output = -1;
#endif

        ~Process() { stop(); }

        void fail(std::string message) {
            std::lock_guard lock(mutex);
            if (!closing && failure.empty()) failure = std::move(message);
        }

        bool launch(const std::vector<std::string>& args) {
#ifdef _WIN32
            SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
            std::array<HANDLE, 3> streams{};
            auto cleanup = [&] { for (HANDLE& stream : streams) close(stream); };
            if (!CreatePipe(&streams[0], &input, &security, 0) ||
                !CreatePipe(&output, &streams[1], &security, 0) ||
                !SetHandleInformation(input, HANDLE_FLAG_INHERIT, 0) ||
                !SetHandleInformation(output, HANDLE_FLAG_INHERIT, 0)) {
                fail("Cannot create strategy pipes"); cleanup(); return false;
            }
            HANDLE standardError = GetStdHandle(STD_ERROR_HANDLE);
            if (!standardError || standardError == INVALID_HANDLE_VALUE ||
                !DuplicateHandle(GetCurrentProcess(), standardError, GetCurrentProcess(), &streams[2], 0, TRUE, DUPLICATE_SAME_ACCESS)) {
                streams[2] = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &security, OPEN_EXISTING, 0, nullptr);
                if (streams[2] == INVALID_HANDLE_VALUE) {
                    streams[2] = nullptr;
                    fail("Cannot connect strategy stderr"); cleanup(); return false;
                }
            }

            // inherit only these streams so a second strategy cannot keep the first strategy's pipes open.
            SIZE_T size = 0;
            InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
            std::vector<std::byte> storage(size);
            auto* attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
            if (!InitializeProcThreadAttributeList(attributes, 1, 0, &size)) {
                fail("Cannot initialize strategy handles"); cleanup(); return false;
            }
            if (!UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, streams.data(),
                                            sizeof(streams), nullptr, nullptr)) {
                DeleteProcThreadAttributeList(attributes);
                fail("Cannot configure strategy handles"); cleanup(); return false;
            }
            STARTUPINFOEXW startup{};
            startup.StartupInfo.cb = sizeof(startup);
            startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
            startup.StartupInfo.hStdInput = streams[0];
            startup.StartupInfo.hStdOutput = streams[1];
            startup.StartupInfo.hStdError = streams[2];
            startup.lpAttributeList = attributes;
            std::wstring command;
            for (const auto& argument : args) {
                if (!command.empty()) command += L' ';
                command += quoted(wide(argument));
            }
            PROCESS_INFORMATION info{};
            BOOL created = CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE,
                EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &startup.StartupInfo, &info);
            DWORD error = GetLastError();
            DeleteProcThreadAttributeList(attributes);
            cleanup();
            if (!created) { fail("Cannot start strategy (Windows error " + std::to_string(error) + ")"); return false; }
            child = info.hProcess;
            // the job owns descendants too, including a Python launcher and its interpreter process.
            job = CreateJobObjectW(nullptr, nullptr);
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) ||
                !AssignProcessToJobObject(job, child) || ResumeThread(info.hThread) == DWORD(-1)) {
                TerminateProcess(child, 1);
                CloseHandle(info.hThread);
                fail("Cannot supervise strategy process"); return false;
            }
            CloseHandle(info.hThread);
#else
            std::array<int, 2> toChild{-1, -1}, fromChild{-1, -1};
            auto cleanup = [&] { for (int& fd : toChild) close(fd); for (int& fd : fromChild) close(fd); };
            if (pipe(toChild.data()) || pipe(fromChild.data())) {
                fail("Cannot create strategy pipes"); cleanup(); return false;
            }
            for (int fd : {toChild[0], toChild[1], fromChild[0], fromChild[1]}) {
                if (fcntl(fd, F_SETFD, FD_CLOEXEC) == -1) {
                    fail("Cannot configure strategy pipes"); cleanup(); return false;
                }
            }
            posix_spawn_file_actions_t files;
            posix_spawnattr_t attributes;
            int status = posix_spawn_file_actions_init(&files);
            if (status) { fail("Cannot initialize process streams"); cleanup(); return false; }
            status = posix_spawnattr_init(&attributes);
            if (status) {
                posix_spawn_file_actions_destroy(&files);
                fail("Cannot initialize process attributes"); cleanup(); return false;
            }
            status = posix_spawn_file_actions_adddup2(&files, toChild[0], STDIN_FILENO);
            if (!status) status = posix_spawn_file_actions_adddup2(&files, fromChild[1], STDOUT_FILENO);
            if (!status) status = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
            if (!status) status = posix_spawnattr_setpgroup(&attributes, 0);
            std::vector<char*> argv;
            for (const auto& argument : args) argv.push_back(const_cast<char*>(argument.c_str()));
            argv.push_back(nullptr);
            if (!status) status = posix_spawnp(&child, argv[0], &files, &attributes, argv.data(), environ);
            posix_spawn_file_actions_destroy(&files);
            posix_spawnattr_destroy(&attributes);
            if (status) { child = -1; fail("Cannot start strategy: " + std::generic_category().message(status)); cleanup(); return false; }
            input = toChild[1]; toChild[1] = -1;
            output = fromChild[0]; fromChild[0] = -1;
            cleanup();
#endif
            return true;
        }

        bool write(std::string_view text) {
            while (!text.empty()) {
#ifdef _WIN32
                DWORD count = 0;
                if (!WriteFile(input, text.data(), static_cast<DWORD>(text.size()), &count, nullptr) || count == 0) return false;
#else
                ssize_t count = ::write(input, text.data(), text.size());
                if (count < 0 && errno == EINTR) continue;
                if (count <= 0) return false;
#endif
                text.remove_prefix(static_cast<std::size_t>(count));
            }
            return true;
        }

        bool read(std::string& line) {
            line.clear();
            // bound an untrusted reply even when a strategy prints an unterminated line.
            while (line.size() < 256) {
                char letter = 0;
#ifdef _WIN32
                DWORD count = 0;
                if (!ReadFile(output, &letter, 1, &count, nullptr) || count != 1) return false;
#else
                ssize_t count = ::read(output, &letter, 1);
                if (count < 0 && errno == EINTR) continue;
                if (count != 1) return false;
#endif
                if (letter == '\n') return true;
                line += letter;
            }
            return false;
        }

        void exchange(Protocol::Init init) {
#ifndef _WIN32
            // a closed child input reports EPIPE on this worker instead of terminating the application.
            sigset_t signals;
            sigemptyset(&signals);
            sigaddset(&signals, SIGPIPE);
            pthread_sigmask(SIG_BLOCK, &signals, nullptr);
#endif
            std::ostringstream greeting;
            Protocol::writeInit(greeting, init);
            if (!write(greeting.str())) fail("Strategy closed its input");
            else while (true) {
                Observation view;
                {
                    std::unique_lock lock(mutex);
                    changed.wait(lock, [&] { return closing || pending.has_value(); });
                    if (closing) break;
                    view = *pending;
                    pending.reset();
                }
                std::ostringstream message;
                if (!Protocol::writeObservation(message, view) || !write(message.str())) {
                    fail("Strategy closed its input"); break;
                }
                std::string line;
                if (!read(line)) { fail("Strategy exited or sent an oversized reply"); break; }
                Time arrival = std::chrono::steady_clock::now();
                std::istringstream incoming(line);
                auto next = Protocol::readAction(incoming);
                if (!next) { fail("Strategy reply has invalid action data"); break; }
                std::lock_guard lock(mutex);
                // only one observation is awaiting a reply, so its tick identifies this action locally.
                reply = next;
                repliedTick = view.tick;
                received = arrival;
            }
            close(input); // EOF lets a cooperative strategy finish its game and exit.
            {
                std::lock_guard lock(mutex);
                done = true;
            }
            changed.notify_all();
        }

        void terminate() {
#ifdef _WIN32
            if (job) TerminateJobObject(job, 1);
            if (child) TerminateProcess(child, 1);
#else
            if (child > 0) kill(-child, SIGKILL);
#endif
        }

        void stop() {
            {
                std::unique_lock lock(mutex);
                closing = true;
                pending.reset();
                changed.notify_all();
                // allow an idle worker to close stdin. interrupt a hung read or write after a short grace period.
                if (!changed.wait_for(lock, std::chrono::milliseconds(100), [&] { return done; })) {
                    terminate();
                }
            }
            if (worker.joinable()) worker.join();
#ifdef _WIN32
            if (child && WaitForSingleObject(child, 100) == WAIT_TIMEOUT) terminate();
            close(job);
            close(child);
#else
            if (child > 0) {
                int status = 0;
                if (waitpid(child, &status, WNOHANG) == 0) {
                    terminate();
                    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
                }
                // stop any descendants that outlived their strategy parent.
                kill(-child, SIGKILL);
                child = -1;
            }
#endif
            close(input);
            close(output);
        }
    };

    StrategyProcess::StrategyProcess() : process(std::make_unique<Process>()) {}
    StrategyProcess::~StrategyProcess() = default;

    bool StrategyProcess::start(std::string_view command, const Protocol::Init& init) {
        process = std::make_unique<Process>();
        auto args = arguments(command);
        if (args.empty()) { process->fail("Enter a program command with balanced quotes"); return false; }
        if (!Protocol::valid(init)) { process->fail("Invalid strategy initialization"); return false; }
        if (!process->launch(args)) { process->stop(); return false; }
        process->done = false;
        process->worker = std::thread([state = process.get(), init] { state->exchange(init); });
        return true;
    }

    void StrategyProcess::stop() { process->stop(); }

    void StrategyProcess::request(const Observation& view) {
        std::lock_guard lock(process->mutex);
        if (process->closing || process->done) return;
        // one in-flight observation and one latest pending observation bound memory for slow strategies.
        process->pending = view;
        process->changed.notify_one();
    }

    Action StrategyProcess::action(std::uint64_t tick, Time cutoff) const {
        std::lock_guard lock(process->mutex);
        if (process->reply && process->repliedTick == tick && process->received <= cutoff) return *process->reply;
        return {};
    }

    std::string StrategyProcess::error() const {
        std::lock_guard lock(process->mutex);
        return process->failure;
    }
}
