from libcpp.pair cimport pair
from libcpp cimport bool

cdef extern from "<boost/container/flat_set.hpp>" namespace "boost::container":
    cdef cppclass flat_set[T]:
        cppclass iterator:
            T& operator*()
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        cppclass reverse_iterator:
            T& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(reverse_iterator) nogil
            bint operator!=(reverse_iterator) nogil
        flat_set() nogil except +
        flat_set(flat_set&) nogil except +
        iterator begin() nogil
        iterator end() nogil
        reverse_iterator rbegin() nogil
        reverse_iterator rend() nogil
        bool empty() nogil
        size_t size() nogil
        size_t max_size() nogil
        size_t capacity() nogil
        void reserve(size_t) nogil
        void shrink_to_fit() nogil
        pair[iterator, bool] insert(T&)
        iterator insert(iterator, T&)
        iterator equal_range(T&)
        pair[iterator, iterator] equal_range(T&)
