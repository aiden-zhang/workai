






#include "dl_timer.h"

DlTimerFactory::DlTimerFactory()
{

}

DlTimerFactory::~DlTimerFactory()
{

}

void DlTimerFactory::registerDlTimerDestructor(DlTimerType type, DlTimerDestructor des)
{
    if(m_map_destructor.find(type) != m_map_destructor.end()) {

    } else {
        m_map_destructor.insert(std::pair<DlTimerType, DlTimerDestructor>(type, des));
    }
}

void DlTimerFactory::registerDlTimerConstructor(DlTimerType type, DlTimerConstructor cons)
{
    if(m_map_constructor.find(type) != m_map_constructor.end()) {

    } else {
        m_map_constructor.insert(std::pair<DlTimerType, DlTimerConstructor>(type, cons));
    }
}

DlTimer* DlTimerFactory::createDlTimer(DlTimerType type)
{
    DlTimer* dl_timer = nullptr;

    m_mutex.lock();
    std::map<DlTimerType, DlTimerConstructor>::iterator cons_iter;
    if((cons_iter = m_map_constructor.find(type)) != m_map_constructor.end()) {
        dl_timer = cons_iter->second();
        if(dl_timer) {
            m_map_dl_timers.insert(std::pair<DlTimer*, DlTimerType>(dl_timer, type));
        }
    }
    m_mutex.unlock();

    return dl_timer;
}

void DlTimerFactory::releaseDlTimer(DlTimer* dl_timer)
{
    std::map<DlTimer*, DlTimerType>::iterator dl_timer_iter;
    m_mutex.lock();
    if((dl_timer_iter = m_map_dl_timers.find(dl_timer)) != m_map_dl_timers.end()) {
        auto type = dl_timer_iter->second;
        if(m_map_destructor.find(type) != m_map_destructor.end()) {
            m_map_destructor[type](dl_timer);
        } else {

        }
    } else {

    }
    m_mutex.unlock();
}


