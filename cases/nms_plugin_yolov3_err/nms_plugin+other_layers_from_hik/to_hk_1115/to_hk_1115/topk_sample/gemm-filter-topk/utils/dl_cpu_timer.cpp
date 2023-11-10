






#include <sys/time.h>
#include "dl_timer.h"


class DlCpuTimer : public DlTimer
{
public:
    DlCpuTimer();
    ~DlCpuTimer();

    void start() override;
    void stop() override;
    int total_count() override;
    float last_elapsed() override;
    float total_elapsed() override;

private:
    struct timeval m_start_time;
};

DlCpuTimer::DlCpuTimer()
{
}

DlCpuTimer::~DlCpuTimer()
{

}

void DlCpuTimer::start()
{
    if(!m_is_on_timing) {
        m_count++;
        m_is_on_timing = true;
        gettimeofday(&m_start_time, nullptr);
    }
}

void DlCpuTimer::stop()
{
    if(m_is_on_timing) {
        m_is_on_timing = false;
        struct timeval cur_time;
        gettimeofday(&cur_time, nullptr);
        m_last_time = (cur_time.tv_sec - m_start_time.tv_sec) * 1000
                + (cur_time.tv_usec - m_start_time.tv_usec) / 1000.0;
        m_total_time += m_last_time;
    }
}

int DlCpuTimer::total_count()
{
    return m_count;
}

float DlCpuTimer::last_elapsed()
{
    return m_last_time;
}

float DlCpuTimer::total_elapsed()
{
    return m_total_time;
}

static DlTimer* createDlCpuTimer()
{
    return new DlCpuTimer();
}

static void releaseDlCpuTimer(DlTimer* dl_timer)
{
    delete dl_timer;
}

REGISTER_DL_TIMER_DES_CONS(DlTimerType_CPU, DlCpuTimer, releaseDlCpuTimer, createDlCpuTimer);
