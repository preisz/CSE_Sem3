/// \file
/// \brief Reference implementation of algo-functions using explicit loops

#include <cmath>   // std::abs
#include <cstddef> // std::size_t
#include <ostream> // std::ostream

#pragma once

namespace iue {

/// \brief reset container to constant value
template <typename C> void set_to_value(C& ctr, typename C::value_type value) {
  for (std::size_t i = 0; i < ctr.size(); ++i) {
    ctr[i] = value;
  }
}

/// \brief set container elements to a range of increasing values
//std::iota Fills the range with sequentially increasing values, starting with value and repetitively evaluating ++value. 
template <typename C> void set_to_sequence(C& ctr, typename C::value_type start) {
  for (std::size_t i = 0; i < ctr.size(); ++i) {
    std::iota(ctr.begin(), ctr.end(), start);
  }
}

/// \brief multiply each element in container by value
//The lambda function modifies each element in-place.
template <typename C> void multiply_with_value(C& ctr, typename C::value_type value) {
  std::for_each(   ctr.begin(), ctr.end(),
            [value](typename C::value_type& element) {element *= value;}    );
}

/// \brief reverse order of elements in container
template <typename C> void reverse_order(C& ctr) {
  std::reverse(ctr.begin(), ctr.end());
}

/// \brief count how many elements fulfill a condition
template <typename C, typename UnaryPredicate>
std::size_t count_fulfills_cond(C& ctr, UnaryPredicate condition) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < ctr.size(); ++i) {
    if (condition(ctr[i])) {
      ++count;
    }
  }
  return count;
}

/// \brief check for identical sequence of values for the first n elements
template <typename C> bool first_n_equal(const C& a, const C& b, std::size_t n) {

  if (a.size() < n || b.size() < n) {
    return false;
  }

  for (std::size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

/// \brief sum all elements in the container
template <typename C> auto sum_of_elements(const C& ctr) {
  typename C::value_type res = 0.0;
  for (std::size_t i = 0; i < ctr.size(); ++i) {
    res += ctr[i];
  }
  return res;
}

/// \brief sum of absolute values of elements in the container
template <typename C> auto abssum_of_elements(C& ctr) {
  typename C::value_type res = 0.0;
  for (std::size_t i = 0; i < ctr.size(); ++i) {
    res += std::abs(ctr[i]);
  }
  return res;
}

/// \brief print all container elements
template <typename C> void print(std::ostream& os, const C& ctr) {
  os << "[ ";
  for (std::size_t i = 0; i < ctr.size(); ++i) {
    os << ctr[i] << " ";
  }
  os << "]" << std::endl;
}

} // namespace iue
