// Copyright (C) 2009-2014 Garth N. Wells, Florian Rathgeber
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-08-09
// Last changed: 2014-05-12

#ifndef __VEC_SET_H
#define __VEC_SET_H

#include <algorithm>
#include <vector>

// This is a set-like data structure. It is not ordered and it is based
// a std::vector. It uses linear search, and can be faster than std::set
// and boost::unordered_set in some cases.

template<typename T>
class vecset {
  public:

    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef typename std::vector<T>::size_type size_type;

    /// Create empty set
    vecset() {}

    /// Create empty set but reserve capacity for n values
    vecset(size_type n) {
      _x.reserve(n);
    }

    /// Copy constructor
    vecset(const vecset<T>& x) : _x(x._x) {}

    /// Destructor
    ~vecset() {}

    /// Find entry in set and return an iterator to the entry
    iterator find(const T& x) {
      return std::find(_x.begin(), _x.end(), x);
    }

    /// Find entry in set and return an iterator to the entry (const)
    const_iterator find(const T& x) const {
      return std::find(_x.begin(), _x.end(), x);
    }

    /// Insert entry
    bool insert(const T& x) {
      if( find(x) == this->end() ) {
        _x.push_back(x);
        return true;
      } else {
        return false;
      }
    }

    /// Insert entries
    template <typename InputIt>
    void insert(const InputIt first, const InputIt last) {
      for (InputIt position = first; position != last; ++position)
      {
        if (std::find(_x.begin(), _x.end(), *position) == _x.end())
          _x.push_back(*position);
      }
    }

    const_iterator begin() const {
      return _x.begin();
    }

    const_iterator end() const {
      return _x.end();
    }

    /// vecset size
    std::size_t size() const {
      return _x.size();
    }

    /// Erase an entry
    void erase(const T& x) {
      iterator p = find(x);
      if (p != _x.end())
        _x.erase(p);
    }

    /// Sort set
    void sort() {
      std::sort(_x.begin(), _x.end());
    }

    /// Clear set
    void clear() {
      _x.clear();
    }

    /// Reserve space for a given number of set members
    void reserve(size_type n) {
      _x.reserve(n);
    }

    /// Set capacity
    size_type capacity() {
      return _x.capacity();
    }

    /// Index the nth entry in the set
    T operator[](size_type n) const {
      return _x[n];
    }

  private:

    std::vector<T> _x;
};

#endif
