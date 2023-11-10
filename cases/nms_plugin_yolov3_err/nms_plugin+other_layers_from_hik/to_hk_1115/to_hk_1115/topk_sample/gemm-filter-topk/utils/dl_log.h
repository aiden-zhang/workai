






#ifndef DL_LOG_H
#define DL_LOG_H

#include <sstream>

typedef enum enDLLoggerSeverity {
    DLLoggerSeverity_Debug     = 0x00000000,
    DLLoggerSeverity_INFO      = 0x00000001,
    DLLoggerSeverity_WARNING   = 0x00000002,
    DLLoggerSeverity_ERROR     = 0x00000003,
    DLLoggerSeverity_FATAL     = 0x00000004
} DLLoggerSeverity;

#define DlLogD DLLogger(__FILE__, __LINE__, DLLoggerSeverity_Debug).log()
#define DlLogI DLLogger(__FILE__, __LINE__, DLLoggerSeverity_INFO).log()
#define DlLogW DLLogger(__FILE__, __LINE__, DLLoggerSeverity_WARNING).log()
#define DlLogE DLLogger(__FILE__, __LINE__, DLLoggerSeverity_ERROR).log()
#define DlLogF DLLogger(__FILE__, __LINE__, DLLoggerSeverity_FATAL).log()


class DLLogger
{
public:
    explicit DLLogger(const char* file, int line, DLLoggerSeverity severity);
    ~DLLogger();

    std::ostream& log();

private:
    const char* m_file;
    int m_line;
    DLLoggerSeverity m_severity;
    std::ostream m_dummy;
    std::ostringstream m_loggger_ostr;
    void *m_logger;
};

void initDlLogger();
void setDlLoggerSeverity(DLLoggerSeverity severity);
DLLoggerSeverity getLoggerSeverityFromString(std::string s);

#ifdef USE_GLOG
void setDlLogParams(const std::string &log_dir, int file_size, int buf_secs = 0);
#endif

void deinitDlLogger();

#endif //DL_LOG_H
