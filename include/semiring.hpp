#ifndef LOG_SEMIRING_HPP 
#define LOG_SEMIRING_HPP 

#include <iostream>
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
class LogType<T>
{
    /*
     * Implement log semiring arithmetic for the given scalar type. 
     */
    private:
        T value; 
        T base; 

    public:
        LogType()
        {
            /*
             * Empty constructor; set to zero. 
             */
            this->value = 0.0;
            this->base = 0.0;
        }

        LogType(T value)
        {
            /*
             * Constructor with specified value; assume base = 10. 
             */
            this->value = value; 
            this->base = 10.0;
        }

        LogType(T value, T base)
        {
            /*
             * Constructor with given value and base. 
             */
            this->value = value; 
            this->base = base; 
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
            return this->value; 
        }

        T base() const 
        {
            /*
             * Return the base. 
             */
            return this->base; 
        }

        T expValue() const 
        {
            /*
             * Return the value in linear (exponentiated) space. 
             */
            return std::pow(this->base, this->value);
        }

        LogType& operator=(const LogType<T>& other)
        {
            /*
             * Assignment operator: assigns both value and base.
             */
            this->value = other.value();
            this->base = other.base();
            return *this;
        }

        LogType& operator=(const T other)
        {
            /*
             * Assignment operator with only value specified (base unchanged). 
             */
            this->value = other;
            return *this; 
        }

        bool operator==(const LogType<T>& other) const
        {
            /*
             * Equality operator.
             */
            return (this->value == other.value() && this->base == other.base());
        }

        bool operator!=(const LogType<T>& other) const
        {
            /*
             * Inequality operator.
             */
            return (this->value != other.value() || this->base != other.base());
        }

        bool operator<(const LogType<T>& other) const
        {
            /*
             * Less-than operator.
             *
             * Raise std::invalid_argument if bases do not match. 
             */
            if (this->base != other.base())
                throw std::invalid_argument("Bases do not match");

            return (this->value < other.value());
        }

        bool operator>(const LogType<T>& other) const 
        {
            /*
             * Greater-than operator.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base != other.base())
                throw std::invalid_argument("Bases do not match");

            return (this->value > other.value());
        }

        bool operator<=(const LogType<T>& other) const
        {
            /*
             * Less-than-or-equal-to operator.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base != other.base())
                throw std::invalid_argument("Bases do not match");

            return (this->value <= other.value());
        }

        bool operator>=(const LogType<T>& other) const
        {
            /*
             * Greater-than-or-equal-to operator.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base != other.base())
                throw std::invalid_argument("Bases do not match");

            return (this->value >= other.value());
        }

        LogType operator+(const LogType<T>& logb) const
        {
            /*
             * Return the result of adding log(b) via logsumexp.
             *
             * Raise std::invalid_argument if bases do not match. 
             */
            if (this->base != logb.base())
                throw std::invalid_argument("Bases do not match");

            return LogType<T>(logsumexp<T>(this->value, logb.value(), this->base), this->base);
        }

        LogType operator*(const LogType<T>& logb) const
        {
            /*
             * Return the result of multiplying by log(b) via addition in log-space.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base != logb.base())
                throw std::invalid_argument("Bases do not match");

            return LogType<T>(this->value + logb.value(), this->base); 
        }

        LogType operator/(const LogType<T>& logb) const
        {
            /*
             * Return the result of dividing by log(b) via subtraction in log-space.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base != logb.base())
                throw std::invalid_argument("Bases do not match");

            return LogType<T>(this->value - logb.value(), this->base);
        }

        LogType& operator+=(const LogType<T>& logb)
        {
            /*
             * In-place addition by log(b) via logsumexp. 
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base != logb.base())
                throw std::invalid_argument("Bases do not match");

            this->value = logsumexp<T>(this->value, logb.value(), this->base);
            return *this;
        }

        LogType& operator*=(const LogType<T>& logb)
        {
            /*
             * In-place multiplication by log(b) via addition in log-space.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base != logb.base())
                throw std::invalid_argument("Bases do not match");

            this->value = this->value + logb.value();
            return *this;
        }

        LogType& operator/=(const LogType<T>& logb)
        {
            /*
             * In-place division by log(b) via subtraction in log-space.
             *
             * Raise std::invalid_argument if bases do not match.
             */
            if (this->base != logb.base())
                throw std::invalid_argument("Bases do not match");

            this->value = this->value - logb.value();
            return *this; 
        }

        /*
         * Friend function declarations for further operator overloads. 
         */
        friend LogType operator+(LogType<T> loga, const LogType<T>& logb);
        friend LogType operator*(LogType<T> loga, const LogType<T>& logb);
        friend LogType operator/(LogType<T> loga, const LogType<T>& logb);
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
