






#include <cuda_runtime.h>
#include <cudnn.h>

#include "dl_timer.h"


class DlGpuTimer : public DlTimer
{
public:
    DlGpuTimer();
    ~DlGpuTimer();

    void start() override;
    void stop() override;
    int total_count() override;
    float last_elapsed() override;
    float total_elapsed() override;

private:
    cudaEvent_t m_evetn_start;
    cudaEvent_t m_evetn_stop;
};

DlGpuTimer::DlGpuTimer()
{
    cudaEventCreate(&m_evetn_start);
    cudaEventCreate(&m_evetn_stop);
}

DlGpuTimer::~DlGpuTimer()
{
    cudaEventDestroy(m_evetn_start);
    cudaEventDestroy(m_evetn_stop);
}

void DlGpuTimer::start()
{
    if(!m_is_on_timing) {
        m_count++;
        m_is_on_timing = true;
        cudaEventRecord(m_evetn_start, nullptr);
    }
}

void DlGpuTimer::stop()
{
    if(m_is_on_timing) {
        m_is_on_timing = false;
        cudaEventRecord(m_evetn_stop, nullptr);
        cudaEventSynchronize(m_evetn_stop);
        cudaEventElapsedTime(&m_last_time, m_evetn_start, m_evetn_stop);
        m_total_time += m_last_time;
    }
}

int DlGpuTimer::total_count()
{
    return m_count;
}

float DlGpuTimer::last_elapsed()
{
    return m_last_time;
}

float DlGpuTimer::total_elapsed()
{
    return m_total_time;
}

static DlTimer* createDLGpuTimer()
{
    return new DlGpuTimer();
}

static void releaseDlGpuTimer(DlTimer* dl_timer)
{
    delete dl_timer;
}

REGISTER_DL_TIMER_DES_CONS(DlTimerType_GPU, DlGpuTimer, releaseDlGpuTimer, createDLGpuTimer);
