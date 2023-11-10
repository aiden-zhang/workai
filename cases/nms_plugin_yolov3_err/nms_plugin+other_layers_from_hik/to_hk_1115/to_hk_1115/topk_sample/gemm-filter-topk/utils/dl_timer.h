






#ifndef DL_TIMER_H
#define DL_TIMER_H

#include <map>
#include <mutex>

typedef enum enDlTimerType
{
    DlTimerType_CPU,
    DlTimerType_GPU
}DlTimerType;

class DlTimer
{
public:
    virtual ~DlTimer() {}
    void reset() {
        m_count = 0;
        m_last_time = 0;
        m_total_time = 0;
        m_is_on_timing = false;
    }
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual int total_count() = 0;
    virtual float last_elapsed() = 0;
    virtual float total_elapsed() = 0;

protected:
    int m_count = 0;
    float m_last_time   = 0.0f;
    float m_total_time  = 0.0f;
    bool m_is_on_timing = false;
};

typedef DlTimer* (*DlTimerConstructor)();
typedef void (*DlTimerDestructor)(DlTimer*);

class DlTimerFactory
{
public:
    virtual ~DlTimerFactory();

    static DlTimerFactory& getInstance() {
        static DlTimerFactory dl_timer_creator;
        return dl_timer_creator;
    }

    void registerDlTimerDestructor(DlTimerType type, DlTimerDestructor des);
    void registerDlTimerConstructor(DlTimerType type, DlTimerConstructor cons);

    DlTimer* createDlTimer(DlTimerType type);
    void releaseDlTimer(DlTimer* dl_timer);

private:
    DlTimerFactory();
    DlTimerFactory(DlTimerFactory&) = delete;
    DlTimerFactory& operator = (const DlTimerFactory&) = delete;

    std::mutex m_mutex;
    std::map<DlTimer*, DlTimerType> m_map_dl_timers = {};
    std::map<DlTimerType, DlTimerDestructor> m_map_destructor = {};
    std::map<DlTimerType, DlTimerConstructor> m_map_constructor = {};
};

#define REGISTER_DL_TIMER_DES_CONS(Type, Name, Des, Cons) \
    __attribute__((constructor)) void Register##Name() { \
        DlTimerFactory::getInstance().registerDlTimerDestructor(Type, Des); \
        DlTimerFactory::getInstance().registerDlTimerConstructor(Type, Cons); \
    }

#endif //DL_TIMER_H
