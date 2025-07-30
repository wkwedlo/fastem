/*
 * EMException.h
 *
 *  Created on: Mar 26, 2017
 *      Author: wkwedlo
 */

#ifndef EM2_EMEXCEPTION_H_
#define EM2_EMEXCEPTION_H_

#include <stdexcept>

using std::runtime_error;

class EMException : public runtime_error{
public:
	EMException(const char *what);
	virtual ~EMException() throw();
};

#endif /* EM2_EMEXCEPTION_H_ */
