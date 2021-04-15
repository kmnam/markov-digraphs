#ifndef LOG_SEMIRING_HPP 
#define LOG_SEMIRING_HPP 

#include <iostream>
#include <sstream>
#include <Eigen/Core>

/*
 * Implementation of log semiring type. 
 * 
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     4/14/2021
 */
template <typename T>
T logsumexp(T loga, T logb, T base = 10.0)
{
    /*
     * Given log(a) and log(b), return log(a + b).
     *
     * The given log-values are assumed to be of the specified base. 
     */
    if (loga > logb) return loga + (std::log1p(std::pow(base, logb - loga)) / std::log(base));
    else             return logb + (std::log1p(std::pow(base, loga - logb)) / std::log(base));
}

template <typename T>
T logdiffexp(T loga, T logb, T base = 10.0)
{
    /*
     * Given log(a) and log(b) with log(a) > log(b), return log(a - b).
     *
     * The given log-values are assumed to be of the specified base.
     */
    if (loga <= logb) throw std::invalid_argument("log(a - b) undefined with log(a) <= log(b)");
    else              return loga - (std::log1p(-std::pow(base, logb - loga)) / std::log(base));
}

template <typename T>
class LogType
{
    /*
     * Implement log semiring arithmetic for the given scalar type. 
     */
    private:
        T value_; 
        T base_; 

    public:
        LogType()
        {
            /*
             * Empty constructor; set to zero. 
             */
            this->value_ = 0.0;
            this->base_ = 0.0;
        }

        LogType(T value)
        {
            /*
             * Constructor with specified value; assume base = 10. 
             */
            this->value_ = value; 
            this->base_ = 10.0;
        }

        LogType(T value, T base)
        {
            /*
             * Constructor with given value and base. 
             */
            this->value_ = value; 
            this->base_ = base; 
        }

        ~LogType()
        {
            /*
             * Trivial destructor. 
             */
        }

        T value() const 
        {
            /*
             * Return the (log) value. 
             */
            return this->value_; 
        }

        T base() const 
        {
            /*
             * Return the base. 
             */
            return this->base_; 
        }

        T expValue() const 
        {
            /*
             * Return the value in linear (exponentiated) space. 
             */
            return std::pow(this->base_, this->value_);
        }

        LogType& operator=(const LogType<T>& logx)
        {
            /*
             * Assignment operator: assigns both value and base.
             */
            this->value_ = logx.value();
            this->base_ = logx.base();
            return *this;
        }

        LogType& operator=(const T logx)
        {
            /*
             * Assignment operator with only value specified (base unchanged).
             *
             * Note that the given scalar value is assumed to lie in log-space. 
             */
            this->value_ = logx;
            return *this; 
        }

        bool operator==(const LogType<T>& other) const
        {
            /*
             * Equality operator.
             */
            return (this->value_ == other.value() && this->base_ == other.base());
        }

        bool operator!=(const LogType<T>& other) const
        {
            /*
             * Inequality operator.
             */
            return (this->value_ != other.value() || this->base_ != other.base());
        }

        bool operator<(const LogType<T>& other) const
        {
            /*
             * Less-than operator.
             *
             * Raise std::invalid_argument if bases do not match. 
             */
            if (this->base_ != other.base())
            {
                std::stringstream ss; 
                ss << "Bases do not match: " << this->base_ << " != " << other.base();
                throw std::invalid_argument(ss.str());
            }
            return (this->value_ < other.value());
        }

        bool operator>(const LogType<T>& other) const 
        {
            /*
             * Greater-than operator.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base_ != other.base())
            {
                std::stringstream ss; 
                ss << "Bases do not match: " << this->base_ << " != " << other.base();
                throw std::invalid_argument(ss.str());
            }
            return (this->value_ > other.value());
        }

        bool operator<=(const LogType<T>& other) const
        {
            /*
             * Less-than-or-equal-to operator.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base_ != other.base())
            {
                std::stringstream ss; 
                ss << "Bases do not match: " << this->base_ << " != " << other.base();
                throw std::invalid_argument(ss.str());
            }
            return (this->value_ <= other.value());
        }

        bool operator>=(const LogType<T>& other) const
        {
            /*
             * Greater-than-or-equal-to operator.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base_ != other.base())
            {
                std::stringstream ss; 
                ss << "Bases do not match: " << this->base_ << " != " << other.base();
                throw std::invalid_argument(ss.str());
            }
            return (this->value_ >= other.value());
        }

        LogType operator+(const LogType<T>& logb) const
        {
            /*
             * Return the result of adding log(b) via logsumexp.
             *
             * Raise std::invalid_argument if bases do not match. 
             */
            if (this->base_ != logb.base())
            {
                std::stringstream ss; 
                ss << "Bases do not match: " << this->base_ << " != " << logb.base();
                throw std::invalid_argument(ss.str());
            }
            return LogType<T>(logsumexp<T>(this->value_, logb.value(), this->base_), this->base_);
        }

        LogType operator-(const LogType<T>& logb) const 
        {
            /*
             * Return the result of subtracting log(b) via logdiffexp. 
             *
             * Raise std::invalid_argument if bases do not match or this->value_
             * is less than log(b). 
             */
            if (this->base_ != logb.base())
            {
                std::stringstream ss; 
                ss << "Bases do not match: " << this->base_ << " != " << logb.base();
                throw std::invalid_argument(ss.str());
            }
            else if (this->value_ <= logb.value())
            {
                throw std::invalid_argument("log(a - b) undefined with log(a) <= log(b)");
            }
            return LogType<T>(logdiffexp<T>(this->value_, logb.value(), this->base_), this->base_);
        }

        LogType operator*(const LogType<T>& logb) const
        {
            /*
             * Return the result of multiplying by log(b) via addition in log-space.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base_ != logb.base())
            {
                std::stringstream ss; 
                ss << "Bases do not match: " << this->base_ << " != " << logb.base();
                throw std::invalid_argument(ss.str());
            }
            return LogType<T>(this->value_ + logb.value(), this->base_); 
        }

        LogType operator/(const LogType<T>& logb) const
        {
            /*
             * Return the result of dividing by log(b) via subtraction in log-space.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base_ != logb.base())
            {
                std::stringstream ss; 
                ss << "Bases do not match: " << this->base_ << " != " << logb.base();
                throw std::invalid_argument(ss.str());
            }
            return LogType<T>(this->value_ - logb.value(), this->base_);
        }

        LogType& operator+=(const LogType<T>& logb)
        {
            /*
             * In-place addition by log(b) via logsumexp. 
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base_ != logb.base())
            {
                std::stringstream ss; 
                ss << "Bases do not match: " << this->base_ << " != " << logb.base();
                throw std::invalid_argument(ss.str());
            }
            this->value_ = logsumexp<T>(this->value_, logb.value(), this->base_);
            return *this;
        }

        LogType& operator-=(const LogType<T>& logb)
        {
            /*
             * In-place subtraction by log(b) via logdiffexp. 
             *
             * Raise std::invalid_argument if bases do not match or this->value_
             * is less than log(b).
             */
            if (this->base_ != logb.base())
            {
                std::stringstream ss; 
                ss << "Bases do not match: " << this->base_ << " != " << logb.base();
                throw std::invalid_argument(ss.str());
            }
            else if (this->value_ <= logb.value())
            {
                throw std::invalid_argument("log(a - b) undefined with log(a) <= log(b)");
            }
            this->value_ = logdiffexp<T>(this->value_, logb.value(), this->base_);
            return *this; 
        }

        LogType& operator*=(const LogType<T>& logb)
        {
            /*
             * In-place multiplication by log(b) via addition in log-space.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base_ != logb.base())
            {
                std::stringstream ss; 
                ss << "Bases do not match: " << this->base_ << " != " << logb.base();
                throw std::invalid_argument(ss.str());
            }
            this->value_ = this->value_ + logb.value();
            return *this;
        }

        LogType& operator/=(const LogType<T>& logb)
        {
            /*
             * In-place division by log(b) via subtraction in log-space.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base_ != logb.base())
            {
                std::stringstream ss; 
                ss << "Bases do not match: " << this->base_ << " != " << logb.base();
                throw std::invalid_argument(ss.str());
            }
            this->value_ = this->value_ - logb.value();
            return *this; 
        }
};

/*
 * Eigen::NumTraits specializations for LogType<T> types, where T is 
 * a boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<N> > type. 
 */
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/eigen.hpp>

using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;

namespace Eigen {

template <unsigned N>
struct NumTraits<LogType<number<mpfr_float_backend<N> > > > : NumTraits<number<mpfr_float_backend<N> > >
{
    typedef LogType<number<mpfr_float_backend<N> > > self_type;
    typedef LogType<number<mpfr_float_backend<N> > > Real; 
    typedef LogType<number<mpfr_float_backend<N> > > NonInteger; 
    typedef LogType<number<mpfr_float_backend<N> > > Nested; 

    enum
    {
        IsComplex = false,
        IsInteger = false,
        ReadCost = 2,
        AddCost = 8,
        MulCost = 16,
        IsSigned = true,
        RequireInitialization = 1,
    };
};

}   // namespace Eigen 

#endif 
