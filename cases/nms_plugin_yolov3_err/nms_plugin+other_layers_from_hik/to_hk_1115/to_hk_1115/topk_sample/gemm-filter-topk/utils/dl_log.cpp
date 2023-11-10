






#include <map>
#include <iostream>

#include "dl_log.h"

#ifdef USE_GLOG
#include <glog/logging.h>
#include <glog/log_severity.h>
#endif

static bool g_logger_initilized = false;
static DLLoggerSeverity g_severity = DLLoggerSeverity_INFO;

DLLogger::DLLogger(const char* file, int line, DLLoggerSeverity severity) : m_dummy(nullptr)
{
    m_file      = file;
    m_line      = line;
    m_severity  = severity;
    m_logger    = nullptr;
}

DLLogger::~DLLogger() {
#ifdef USE_GLOG
    if (nullptr != m_logger) {
        delete static_cast<google::LogMessage*>(m_logger);
    }
#else
    if(g_logger_initilized && m_severity >= g_severity) {
        std::cout << m_loggger_ostr.str() << std::endl;
        if(DLLoggerSeverity_FATAL == m_severity) {
            abort();
        }
    }
#endif
}

std::ostream& DLLogger::log()
{
    if(!g_logger_initilized ||  m_severity < g_severity)
    {
        return m_dummy;
    }

#ifdef USE_GLOG
    int google_severity = m_severity + google::GLOG_INFO - LoggerSeverity_INFO;
    google_severity = (google_severity >= google::GLOG_INFO ? google_severity : google::GLOG_INFO);
    m_logger = new google::LogMessage(m_file, m_line, google_severity);
    return static_cast<google::LogMessage*>(m_logger)->stream();
#else
    m_loggger_ostr << m_file << ":" << m_line << " ";
    return m_loggger_ostr;
#endif
}


void initDlLogger()
{
    if(g_logger_initilized) {
        return;
    }

#ifdef USE_GLOG
    google::InitGoogleLogging("DL");
    google::SetStderrLogging(google::GLOG_INFO);
#endif

    g_logger_initilized = true;
}

void setDlLoggerSeverity(DLLoggerSeverity severity)
{
    g_severity = severity;
}


static const std::map<std::string, DLLoggerSeverity> g_logger_serverity_map = {
    {"debug",   DLLoggerSeverity_Debug},
    {"info",    DLLoggerSeverity_INFO},
    {"warning", DLLoggerSeverity_WARNING},
    {"error",   DLLoggerSeverity_ERROR},
    {"fatal",   DLLoggerSeverity_FATAL}
};

DLLoggerSeverity getLoggerSeverityFromString(std::string s)
{
    DLLoggerSeverity ret = DLLoggerSeverity_INFO;
    if(g_logger_serverity_map.find(s) != g_logger_serverity_map.end()) {
        ret = g_logger_serverity_map.at(s);
    } else {
        DlLogE << "Unknown logger level-->" << s;
    }

    return ret;
}

#ifdef USE_GLOG
void setDlLogParams(const std::string &log_dir, int file_size, int buf_secs)
{
    fLS::FLAGS_log_dir = log_dir.c_str();
    fLI::FLAGS_logbufsecs = buf_secs;
    fLI::FLAGS_max_log_size = file_size;
}
#endif

void deinitDlLogger()
{
    if(g_logger_initilized) {

#ifdef USE_GLOG
    google::ShutdownGoogleLogging();
#endif
        g_logger_initilized = false;
    }
}
