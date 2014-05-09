from libcpp cimport bool

cdef extern from "vecset.h":
    cdef cppclass vecset[T]:
        cppclass iterator:
            T& operator*()
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        cppclass const_iterator:
            T& operator*()
            const_iterator operator++() nogil
            const_iterator operator--() nogil
            bint operator==(const_iterator) nogil
            bint operator!=(const_iterator) nogil
        vecset() nogil except +
        vecset(int) nogil except +
        vecset(vecset&) nogil except +
        const_iterator find(T&) nogil
        bool insert(T&)
        void insert(const_iterator, const_iterator)
        const_iterator begin() nogil
        const_iterator end() nogil
        size_t size() nogil
        void erase(T&) nogil
        void sort() nogil
        void clear() nogil
        void reserve(int) nogil
        int capacity() nogil
        T operator[](size_t) nogil
