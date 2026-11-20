//
// Created by wkwedlo on 15.07.25.
//

#ifndef CLUST_LIKWID_H
#define CLUST_LIKWID_H

#ifdef LIKWID
#include <likwid.h>

extern inline void __likwid_markerInit() {
    likwid_markerInit();
}

extern inline void __likwid_markerThreadInit() {
    likwid_markerThreadInit();
}
extern inline void __likwid_markerClose() {
    likwid_markerClose();
}

extern inline void __likwid_markerStartRegion(const char *name) {
    likwid_markerStartRegion(name);
}

extern inline void __likwid_markerStopRegion(const char *name) {
    likwid_markerStopRegion(name);
}

extern inline void __likwid_markerResetRegion(const char *name) {
    likwid_markerResetRegion(name);
}
#else
    extern inline void __likwid_markerInit() {}
    extern inline void __likwid_markerThreadInit() {}
    extern inline void __likwid_markerClose() {}
    extern inline void __likwid_markerStartRegion(const char *name) {}
    extern inline void __likwid_markerStopRegion(const char *name) {}
    extern inline void __likwid_markerResetRegion(const char *name) {}
#endif

#endif //CLUST_LIKWID_H
