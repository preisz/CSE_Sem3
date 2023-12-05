/// \file
/// \brief Distribution class (header-only)

#pragma once

/// \todo add standard library headers as needed
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <typeinfo> //check types

#include <vector>
#include <list>
#include <deque>

#include "VecN.hpp"
namespace ex22 {
    using std::sqrt;

/// \todo Task 1: Implement a type "Distribution" with the following properties:
/// - class template with a single template type parameter T
/// - two public data member 'mean' and 'stddev' of type T
/// - public member typedef "value_type" which aliases the template parameter
/// - two constructors:
///   - Distribution(T mean, T stddev) which sets the two member variables
///   - Distribution(const std::vector<T> &data) which calculates
///     the two member variables from the samples in the container

/*
template <class T>
struct Distribution{
    T mean;
    T stddev;

    using value_type = T; // Public type alias

    Distribution(value_type mean, value_type stddev) : mean(mean), stddev(stddev) {} // Constructor1
    
    Distribution(const std::vector<T>& vec) { // Constructor2 from a std::vector, calculates mean and stddev
        if (vec.empty()) {throw std::logic_error("Cannot calculate mean and stddev from an empty vector");}
        mean = 0; stddev = 0; //set to 0 for the start

        for (const value_type& value : vec){mean = mean + value;} //add up
        mean = static_cast<value_type>( mean / vec.size() ); //divide by length of vector to ge the mean

        value_type squaredDifferences(0);
        for (const value_type& value : vec) { //iterate & sum the squared differences
            squaredDifferences = squaredDifferences + ( (value - mean)*(value -mean)  );
        }
        //if(typeid(value_type) != typeid(VecN))
        //    stddev = std::sqrt(squaredDifferences / static_cast<T>(vec.size())); // Calculate standard deviation
        //else
        stddev = sqrt(squaredDifferences / vec.size());

    }
};

*/


/// \todo Task 2: Extend the construction mechanism:
/// - change the constructor "Distribution(const std::vector<T>&)" so it
///   accepts any sequential container from the standard library
/// - the template type parameter T is still deduced automatically

template <typename T, template <typename...> class CONTAINER = std::vector> //this version allows only providing valuetype T
struct Distribution{
    T mean;
    T stddev;

    using value_type = T; // Public type alias

    Distribution(value_type mean, value_type stddev) : mean(mean), stddev(stddev) {} // Constructor1
    
    Distribution(const CONTAINER<T>& vec) { // Constructor2 from a container
        if (vec.empty()) {throw std::logic_error("Cannot calculate mean and stddev from an empty container");}
        mean = 0; stddev = 0; //set to 0 for the start

        for (const value_type& value : vec){mean = mean + value;} //add up
        mean = static_cast<value_type>( mean / vec.size() ); //divide by length of vector to ge the mean

        value_type squaredDifferences(0);
        for (const value_type& value : vec) { //iterate & sum the squared differences
            squaredDifferences = squaredDifferences + ( (value - mean)*(value -mean)  );
        }
        stddev = sqrt(squaredDifferences / vec.size());

    }

};

} // namespace ex22