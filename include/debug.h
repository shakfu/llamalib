#ifndef DEBUG_H
#define DEBUG_H

// common stdlibs
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

// colors
#define COLOR_RESET         "\033[m"
#define COLOR_BOLD          "\033[1m"
#define COLOR_RED           "\033[31m"
#define COLOR_GREEN         "\033[32m"
#define COLOR_YELLOW        "\033[33m"
#define COLOR_BLUE          "\033[34m"
#define COLOR_MAGENTA       "\033[35m"
#define COLOR_CYAN          "\033[36m"
#define COLOR_BOLD_RED      "\033[1;31m"
#define COLOR_BOLD_GREEN    "\033[1;32m"
#define COLOR_BOLD_YELLOW   "\033[1;33m"
#define COLOR_BOLD_BLUE     "\033[1;34m"
#define COLOR_BOLD_MAGENTA  "\033[1;35m"
#define COLOR_BOLD_CYAN     "\033[1;36m"

// debugging
#define debug(M, ...) fprintf(stderr, "[" COLOR_BOLD_CYAN "DEBUG" COLOR_RESET "] %s:%d: " M "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define clean_errno() (errno == 0 ? "None" : strerror(errno))
#define log_err(M, ...) fprintf(stderr, "[" COLOR_BOLD_RED "ERROR" COLOR_RESET "] (%s:%d: errno: %s) " M "\n", __FILE__, __LINE__, clean_errno(), ##__VA_ARGS__)
#define log_warn(M, ...) fprintf(stderr, "[" COLOR_BOLD_YELLOW "WARN" COLOR_RESET "]  (%s:%d: errno: %s) " M "\n", __FILE__, __LINE__, clean_errno(), ##__VA_ARGS__)
#define log_info(M, ...) fprintf(stderr, "[" COLOR_BOLD_GREEN "INFO" COLOR_RESET "]  (%s:%d) " M "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#endif // DEBUG_H
