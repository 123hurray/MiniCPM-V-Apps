#pragma once

#include <android/log.h>

void diagnostic_log_set_path(const char * path);
void diagnostic_log_install_crash_handlers();
void diagnostic_log_write(int priority, const char * tag, const char * message);
void diagnostic_log_printf(int priority, const char * tag, const char * format, ...)
    __attribute__((format(printf, 3, 4)));

