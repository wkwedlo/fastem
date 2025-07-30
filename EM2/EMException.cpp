/*
 * EMException.cpp
 *
 *  Created on: Mar 26, 2017
 *      Author: wkwedlo
 */

#include "EMException.h"

EMException::EMException(const char *what) : runtime_error(what) {
	// TODO Auto-generated constructor stub

}

EMException::~EMException() throw() {

}

