//
// Created by wkwedlo on 31.10.24.
//

#ifndef CLUST_ITAC_H
#define CLUST_ITAC_H

#ifdef INTEL_ITAC
#include <VT.h>
    extern inline void _vt_classdef(const char *name,int *classID) {
        VT_classdef(name,classID);
    }

    extern inline void _vt_funcdef(const char *name,int classID,int *regionID) {
        VT_funcdef(name,classID,regionID);
    }
    extern inline void _vt_begin(const int regionID) {
        VT_begin(regionID);
    }
    extern inline void _vt_end(const int regionID) {
        VT_end(regionID);
    }

#else
    extern inline void _vt_classdef(const char *name,int *classID) {*classID=0;}
    extern inline void _vt_funcdef(const char *name,int classID,int *regionID) {*regionID=0;}
    extern inline void _vt_begin(const int regionID) {}
    extern inline void _vt_end(const int regionID) {}
#endif
#endif //CLUST_ITAC_H
